import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

x, y = None, None
model = None
reader = None

def inicializar_modelos():
    global model, reader
    if model is None:
        model = YOLO('yolov8n.pt')
    if reader is None:
        reader = easyocr.Reader(['en'])  # 'en' pra inglês, muda se precisar

def encontrar_com_ocr(searching, frame):
    """Tenta encontrar texto na tela"""
    global reader

    results = reader.readtext(frame)

    for (bbox, text, confidence) in results:
        if searching.lower() in text.lower():
            # Calcula o centro do texto encontrado
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]

            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))

            print(f"✓ Encontrado (OCR): {text}")
            print(f"  Coordenadas: ({center_x}, {center_y})")
            print(f"  Confiança: {confidence:.2f}")

            return {
                'encontrado': True,
                'tipo': 'OCR',
                'texto': text,
                'x': center_x,
                'y': center_y,
                'confianca': confidence
            }

    return None

def encontrar_com_yolo(searching, frame):
    """Tenta encontrar objeto com YOLO"""
    global model

    results = model(frame)
    detections = results[0]

    for detection in detections.boxes:
        x1, y1, x2, y2 = detection.xyxy[0]
        confidence = detection.conf[0]
        class_id = int(detection.cls[0])
        class_name = model.names[class_id]

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        if searching.lower() in class_name.lower():
            print(f"✓ Encontrado (YOLO): {class_name}")
            print(f"  Coordenadas: ({center_x}, {center_y})")
            print(f"  Confiança: {confidence:.2f}")

            return {
                'encontrado': True,
                'tipo': 'YOLO',
                'classe': class_name,
                'x': center_x,
                'y': center_y,
                'confianca': float(confidence)
            }

    return None

def encontrar_elemento(searching):
    """Tenta OCR primeiro, se não achar tenta YOLO"""
    global x, y

    inicializar_modelos()

    # Captura screenshot
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Tenta OCR primeiro
    resultado_ocr = encontrar_com_ocr(searching, frame)
    if resultado_ocr:
        x = resultado_ocr['x']
        y = resultado_ocr['y']
        return resultado_ocr

    # Se não achou com OCR, tenta YOLO
    resultado_yolo = encontrar_com_yolo(searching, frame)
    if resultado_yolo:
        x = resultado_yolo['x']
        y = resultado_yolo['y']
        return resultado_yolo

    print(f"✗ Não encontrado: {searching}")
    return {'encontrado': False}

# Uso
if __name__ == "__main__":
    searching = input("O que você quer procurar?\n")
    resultado = encontrar_elemento(searching)

    if resultado['encontrado']:
        pyautogui.moveTo(x, y, duration=0.5)
        print("Finalizado")
