from ultralytics import YOLO
import cv2

# Carregar modelo treinado
model = YOLO("runs/detect/train/weights/best.pt")

# Abrir webcam
cap = cv2.VideoCapture(2)

# Aumentar resolução
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # ou 1920
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # ou 1080

if not cap.isOpened():
    print("Erro ao abrir câmera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame.")
        break

    # Rodar YOLO
    results = model(frame, conf=0.5)

    # Anotar o frame
    annotated_frame = results[0].plot()

    # Mostrar
    cv2.imshow("YOLO Webcam", annotated_frame)

    # Sair com Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
