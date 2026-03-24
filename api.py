import os
import glob
import torch
import uvicorn
import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

        # in_channels: например, 1024
        # feature_map_size: размер картинки на дне (например, 10x10)

        self.flatten_dim = feature_map_size * feature_map_size  # 10*10 = 100 токенов
        self.embed_dim = in_channels

        # Позиционное кодирование (чтобы сеть знала, где какой кусочек находится)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.flatten_dim, self.embed_dim))

        # Сам трансформер (Encoder часть)
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

        # Рассчитываем размер на "дне".
        # Мы делаем 4 пулинга (деление на 2): 160 -> 80 -> 40 -> 20 -> 10.
        self.bottleneck_size = img_size // 16

        # --- ENCODER (CNN) ---
        self.dconv_down1 = DoubleConv(in_channels, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512) # Выход здесь пойдет в трансформер

        self.maxpool = nn.MaxPool2d(2)

        # Переходник перед трансформером (увеличим каналы до 1024)
        self.conv_embed = nn.Conv2d(512, 1024, kernel_size=1)

        # --- BOTTLENECK (TRANSFORMER) ---
        # Вместо обычных сверток тут стоит Self-Attention
        self.transformer = TransformerBottleneck(
            in_channels=1024,
            feature_map_size=self.bottleneck_size, # 10
            num_heads=8,      # Количество "голов" внимания
            num_layers=3      # Глубина трансформера
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # --- DECODER (CNN) ---
        # Возвращаемся к сверткам для восстановления разрешения
        self.dconv_up4 = DoubleConv(1024 + 512, 512)
        self.dconv_up3 = DoubleConv(512 + 256, 256)
        self.dconv_up2 = DoubleConv(256 + 128, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # --- Спуск (CNN) ---
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4) # [Batch, 512, 10, 10]

        # --- Трансформер ---
        # Подготовка каналов (512 -> 1024)
        x = self.conv_embed(x) # [Batch, 1024, 10, 10]

        # Магия внимания
        x = self.transformer(x) # [Batch, 1024, 10, 10]

        # --- Подъем (CNN) ---
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
IMAGE_SIZE = 160
MODEL_PATH = "best_model_trans_unet.pth"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Инициализация приложения FastAPI
app = FastAPI(title="Brain Tumor Segmentation API", description="API для инференса TransUNet на 3D МРТ")

# Глобальная переменная для модели
model = None

@app.on_event("startup")
def load_model():
    """Загрузка модели в память при старте сервера"""
    global model
    print(f"Загрузка модели на {DEVICE}...")
    model = TransUNet(in_channels=4, out_channels=1, img_size=IMAGE_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Модель успешно загружена!")

# Формат входных данных для API
class InferenceRequest(BaseModel):
    directory_path: str

# ==========================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def get_file_path(path, modality_tokens, exclude_tokens=None):
    """Поиск файла модальности (взято из вашего Dataloader)"""
    files = glob.glob(os.path.join(path, "*.nii*"))
    for f in files:
        fname = os.path.basename(f).lower()
        if any(token in fname for token in modality_tokens):
            if exclude_tokens and any(ex in fname for ex in exclude_tokens):
                continue
            return f
    return None

def normalize_slice(slice_2d):
    """Нормализация как при обучении"""
    mask = slice_2d > 0
    if mask.sum() > 0:
        mean = slice_2d[mask].mean()
        std = slice_2d[mask].std()
        slice_2d = (slice_2d - mean) / (std + 1e-8)
        slice_2d[~mask] = 0
    return slice_2d

# ==========================================
# 4. ОСНОВНОЙ ЭНДПОИНТ (API)
# ==========================================
@app.post("/predict")
def predict_segmentation(request: InferenceRequest):
    data_dir = request.directory_path

    if not os.path.isdir(data_dir):
        raise HTTPException(status_code=400, detail="Указанная директория не существует.")

    # 1. Ищем нужные файлы
    flair_p = get_file_path(data_dir, ['flair'])
    t2_p    = get_file_path(data_dir,['t2'])
    t1ce_p  = get_file_path(data_dir,['t1ce', 't1c'])
    t1_p    = get_file_path(data_dir, ['t1'], exclude_tokens=['ce', 't1c'])

    if not all([flair_p, t2_p, t1_p, t1ce_p]):
        raise HTTPException(status_code=400, detail="В папке отсутствуют все 4 необходимые модальности (FLAIR, T1, T1c, T2).")

    try:
        # 2. Читаем 3D объемы (SimpleITK возвращает массивы в формате [Depth, H, W])
        print("Чтение данных...")
        flair_img = sitk.ReadImage(flair_p)

        vol_flair = sitk.GetArrayFromImage(flair_img)
        vol_t1    = sitk.GetArrayFromImage(sitk.ReadImage(t1_p))
        vol_t1ce  = sitk.GetArrayFromImage(sitk.ReadImage(t1ce_p))
        vol_t2    = sitk.GetArrayFromImage(sitk.ReadImage(t2_p))

        depth, H, W = vol_flair.shape

        # 3. Подготовка пустого 3D массива для результата
        pred_volume = np.zeros((depth, H, W), dtype=np.uint8)

        # Вычисляем параметры кропа (как в датасете)
        cx, cy = H // 2, W // 2
        sz = IMAGE_SIZE // 2
        x1, x2 = max(0, cx - sz), min(H, cx + sz)
        y1, y2 = max(0, cy - sz), min(W, cy + sz)

        print(f"Запуск инференса (всего срезов: {depth})...")

        # 4. Послойный инференс
        with torch.no_grad():
            for z in range(depth):
                # Достаем 2D срезы
                s_flair = vol_flair[z]
                s_t1    = vol_t1[z]
                s_t1ce  = vol_t1ce[z]
                s_t2    = vol_t2[z]

                # Пропускаем полностью пустые срезы для скорости
                if s_flair.max() == 0 and s_t2.max() == 0:
                    continue

                # Стек в (4, H, W)
                img_2d = np.stack([s_flair, s_t1, s_t1ce, s_t2], axis=0)

                # Кроп
                img_2d_cropped = img_2d[:, x1:x2, y1:y2]

                # Паддинг если нужно
                pad_h = max(0, IMAGE_SIZE - img_2d_cropped.shape[1])
                pad_w = max(0, IMAGE_SIZE - img_2d_cropped.shape[2])

                if pad_h > 0 or pad_w > 0:
                    img_2d_cropped = np.pad(img_2d_cropped, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')

                # Нормализация
                for i in range(4):
                    img_2d_cropped[i] = normalize_slice(img_2d_cropped[i])

                # Конвертация в тензор и инференс
                tensor_input = torch.from_numpy(img_2d_cropped).float().unsqueeze(0).to(DEVICE)
                output = model(tensor_input)

                # Сигмоида + порог (Бинаризация маски)
                prob = torch.sigmoid(output).cpu().numpy()[0, 0]
                mask_2d = (prob > 0.5).astype(np.uint8)

                # Обратный паддинг (отрезаем лишнее)
                if pad_h > 0 or pad_w > 0:
                    mask_2d = mask_2d[:mask_2d.shape[0]-pad_h, :mask_2d.shape[1]-pad_w]

                # Вставка обратно в 3D объем
                pred_volume[z, x1:x2, y1:y2] = mask_2d

        # 5. Сохранение результата
        output_filename = "predicted_mask.nii"
        output_path = os.path.join(data_dir, output_filename)

        # Создаем NIfTI изображение
        out_sitk = sitk.GetImageFromArray(pred_volume)

        # КРИТИЧНО ВАЖНО: копируем геометрию снимка из исходного файла
        out_sitk.CopyInformation(flair_img)

        sitk.WriteImage(out_sitk, output_path)

        return {"status": "success", "message": "Сегментация завершена", "output_file": output_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
