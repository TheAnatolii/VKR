# Brain Tumor Segmentation API (Docker)

## Запуск через Docker Compose

1. Положите файл весов модели в `models/best_model_trans_unet.pth`.
2. Запустите сборку и контейнер:

```bash
docker compose up --build -d
```

3. API будет доступно по адресу:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

## Пример запроса

Эндпоинт `/predict` принимает 4 файла MRI через `multipart/form-data`:

- `flair`
- `t1`
- `t1ce`
- `t2`

В ответ API возвращает файл `predicted_mask.nii`.

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "flair=@./test_api_segm/Brats18_2013_17_1_flair.nii" \
  -F "t1=@./test_api_segm/Brats18_2013_17_1_t1.nii" \
  -F "t1ce=@./test_api_segm/Brats18_2013_17_1_t1ce.nii" \
  -F "t2=@./test_api_segm/Brats18_2013_17_1_t2.nii" \
  --output predicted_mask.nii
```

В Swagger UI можно просто выбрать эти 4 файла вручную и нажать `Execute`.

## Остановка

```bash
docker compose down
```
