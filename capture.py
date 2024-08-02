import cv2
import math
import torch
from ultralytics import YOLO

def capture_image():
    cap = cv2.VideoCapture(0)
    detect = YOLO('./weights/yolov8l.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detect.to(device)
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        results = detect(img)
        for res in results:
            boxes = res.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if cls == 0 and conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  
                    crop = img[y1:y2, x1:x2]         
            cv2.imshow('img', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            name = input('Enter name:')
            return crop,name
