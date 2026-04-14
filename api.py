import os
import tempfile
import torch
import uvicorn
import numpy as np
import SimpleITK as sitk
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import torch.nn as nn


# ==========================================
# 1. КЛАССЫ МОДЕЛИ
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels, feature_map_size, num_heads=8, num_layers=4, hidden_dim=2048):
        super().__init__()
        
        self.flatten_dim = feature_map_size * feature_map_size
        self.embed_dim = in_channels
        
        self.pos_embedding = nn.Parameter(torch.randn(1, self.flatten_dim, self.embed_dim))
        
        # Encoder часть
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x

class TransUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, img_size=160):
        super().__init__()
        
        self.bottleneck_size = img_size // 16 
        
        # --- ENCODER ---
        self.dconv_down1 = DoubleConv(in_channels, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        
        self.conv_embed = nn.Conv2d(512, 1024, kernel_size=1) 

        # --- BOTTLENECK ---
        self.transformer = TransformerBottleneck(
            in_channels=1024, 
            feature_map_size=self.bottleneck_size,
            num_heads=8,
            num_layers=3
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        # --- DECODER ---
        self.dconv_up4 = DoubleConv(1024 + 512, 512)
        self.dconv_up3 = DoubleConv(512 + 256, 256)
        self.dconv_up2 = DoubleConv(256 + 128, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x) 
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x) 
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x) 
        x = self.maxpool(conv3)   
        
        conv4 = self.dconv_down4(x) 
        x = self.maxpool(conv4)
        

        x = self.conv_embed(x)
        
        x = self.transformer(x)
        
        
        x = self.upsample(x)        
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)   
        
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        return out


# ==========================================
# 2. НАСТРОЙКИ
# ==========================================
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "160"))
MODEL_PATH = os.getenv("MODEL_PATH", "best_model_trans_unet.pth")
DEVICE = os.getenv("DEVICE", "cpu")


app = FastAPI(title="Brain Tumor Segmentation API", description="API для инференса TransUNet на 3D МРТ")

model = None

@app.on_event("startup")
def load_model():
    global model
    print(f"Загрузка модели на {DEVICE}...")
    model = TransUNet(in_channels=4, out_channels=1, img_size=IMAGE_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Модель успешно загружена!")


# ==========================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def normalize_slice(slice_2d):
    """Нормализация"""
    mask = slice_2d > 0
    if mask.sum() > 0:
        mean = slice_2d[mask].mean()
        std = slice_2d[mask].std()
        slice_2d = (slice_2d - mean) / (std + 1e-8)
        slice_2d[~mask] = 0
    return slice_2d


async def save_upload_file(upload: UploadFile, destination: str):
    """Сохраняет загруженный файл на диск без загрузки всего содержимого в память."""
    with open(destination, "wb") as buffer:
        while chunk := await upload.read(1024 * 1024):
            buffer.write(chunk)


def cleanup_file(path: str):
    """Удаляет временный файл после отправки ответа."""
    if os.path.exists(path):
        os.remove(path)


# ==========================================
# 4. ОСНОВНОЙ ЭНДПОИНТ (API)
# ==========================================
@app.post("/predict")
async def predict_segmentation(
    background_tasks: BackgroundTasks,
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            flair_path = os.path.join(temp_dir, flair.filename or "flair.nii.gz")
            t1_path = os.path.join(temp_dir, t1.filename or "t1.nii.gz")
            t1ce_path = os.path.join(temp_dir, t1ce.filename or "t1ce.nii.gz")
            t2_path = os.path.join(temp_dir, t2.filename or "t2.nii.gz")

            await save_upload_file(flair, flair_path)
            await save_upload_file(t1, t1_path)
            await save_upload_file(t1ce, t1ce_path)
            await save_upload_file(t2, t2_path)

            print("Чтение загруженных данных...")
            flair_img = sitk.ReadImage(flair_path)

            vol_flair = sitk.GetArrayFromImage(flair_img)
            vol_t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
            vol_t1ce = sitk.GetArrayFromImage(sitk.ReadImage(t1ce_path))
            vol_t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))

            depth, H, W = vol_flair.shape
            pred_volume = np.zeros((depth, H, W), dtype=np.uint8)

            cx, cy = H // 2, W // 2
            sz = IMAGE_SIZE // 2
            x1, x2 = max(0, cx - sz), min(H, cx + sz)
            y1, y2 = max(0, cy - sz), min(W, cy + sz)

            print(f"Запуск инференса (всего срезов: {depth})...")

            with torch.no_grad():
                for z in range(depth):
                    s_flair = vol_flair[z]
                    s_t1 = vol_t1[z]
                    s_t1ce = vol_t1ce[z]
                    s_t2 = vol_t2[z]

                    if s_flair.max() == 0 and s_t2.max() == 0:
                        continue

                    img_2d = np.stack([s_flair, s_t1, s_t1ce, s_t2], axis=0)
                    img_2d_cropped = img_2d[:, x1:x2, y1:y2]

                    pad_h = max(0, IMAGE_SIZE - img_2d_cropped.shape[1])
                    pad_w = max(0, IMAGE_SIZE - img_2d_cropped.shape[2])

                    if pad_h > 0 or pad_w > 0:
                        img_2d_cropped = np.pad(img_2d_cropped, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')

                    for i in range(4):
                        img_2d_cropped[i] = normalize_slice(img_2d_cropped[i])

                    tensor_input = torch.from_numpy(img_2d_cropped).float().unsqueeze(0).to(DEVICE)
                    output = model(tensor_input)

                    prob = torch.sigmoid(output).cpu().numpy()[0, 0]
                    mask_2d = (prob > 0.5).astype(np.uint8)

                    if pad_h > 0 or pad_w > 0:
                        mask_2d = mask_2d[:mask_2d.shape[0] - pad_h, :mask_2d.shape[1] - pad_w]

                    pred_volume[z, x1:x2, y1:y2] = mask_2d

            temp_output = tempfile.NamedTemporaryFile(suffix=".nii", delete=False)
            output_path = temp_output.name
            temp_output.close()
            out_sitk = sitk.GetImageFromArray(pred_volume)
            out_sitk.CopyInformation(flair_img)
            sitk.WriteImage(out_sitk, output_path)
            background_tasks.add_task(cleanup_file, output_path)

            return FileResponse(
                output_path,
                media_type="application/octet-stream",
                filename="predicted_mask.nii",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
