import pytesseract
pytesseract.pytesseract.pytesseract_cmd = r'C:\Users\igor.batista\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


import cv2
import threading
import queue
import time
from ultralytics import YOLO

# Inputs
search_text = input("O que você quer buscar? (ex: 'placa', 'número'): ").strip()
camera_id = int(input("Qual câmera usar? (0 para padrão): "))

# Configurações
frame_queue = queue.Queue(maxsize=3)
results_queue = queue.Queue()
running = True

# Carrega modelo YOLO
model = YOLO('yolov8n.pt')

def captura_camera():
    """Captura frames da câmera"""
    global running
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    start_time = time.time()

    while running:
        ret, frame = cap.read()
        if ret:
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                pass

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1:
                fps = frame_count / elapsed
                print(f"[CÂMERA] FPS: {fps:.1f}")
                frame_count = 0
                start_time = time.time()

    cap.release()

def processa_ocr():
    """Roda OCR em paralelo"""
    global running
    while running:
        try:
            frame = frame_queue.get(timeout=1)

            # Converte pra escala de cinza pra melhorar OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Roda Tesseract
            text = pytesseract.image_to_string(gray)

            if search_text.lower() in text.lower():
                results_queue.put({
                    'tipo': 'OCR',
                    'encontrado': True,
                    'texto': text[:100]
                })
                print(f"✓ [OCR] Encontrado '{search_text}'")
        except queue.Empty:
            pass
        except Exception as e:
            print(f"✗ [OCR] Erro: {e}")

def processa_yolo():
    """Roda YOLO em paralelo"""
    global running
    while running:
        try:
            frame = frame_queue.get(timeout=1)

            # Roda YOLO
            results = model(frame, verbose=False)

            # Extrai classes detectadas
            for result in results:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)]
                    if search_text.lower() in class_name.lower():
                        results_queue.put({
                            'tipo': 'YOLO',
                            'encontrado': True,
                            'classe': class_name
                        })
                        print(f"✓ [YOLO] Detectado '{class_name}'")
        except queue.Empty:
            pass
        except Exception as e:
            print(f"✗ [YOLO] Erro: {e}")

def main():
    global running

    print(f"\n🎥 Iniciando com câmera {camera_id}")
    print(f"🔍 Buscando por: '{search_text}'")
    print(f"⏱️  Mantendo 21+ fps\n")

    # Inicia threads
    t_camera = threading.Thread(target=captura_camera, daemon=True)
    t_ocr = threading.Thread(target=processa_ocr, daemon=True)
    t_yolo = threading.Thread(target=processa_yolo, daemon=True)

    t_camera.start()
    t_ocr.start()
    t_yolo.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Parando...")
        running = False
        t_camera.join(timeout=2)
        t_ocr.join(timeout=2)
        t_yolo.join(timeout=2)

if __name__ == "__main__":
    main()
