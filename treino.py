from ultralytics import YOLO

# 1. Carrega o modelo base (pré-treinado)
model = YOLO("yolov8n.pt")  # versão pequena e rápida

# 2. Treina o modelo usando seu dataset
model.train(
    data="dataset/data.yaml",   # caminho para seu data.yaml
    epochs=50,                  # pode alterar depois
    imgsz=640,                  # tamanho das imagens
    batch=8,                    # ajuste se der erro de RAM
    device="0"                    # 0 = GPU, "cpu" se não tiver GPU
)
