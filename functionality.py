import torch
from source.CLIP.clip import clip
from PIL import Image
import cv2
from ultralytics import YOLO
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = clip.load("./weights/ViT-L-14.pt", device=device)
detect = YOLO('./weights/yolov8l.pt')
detect.to(device)

def extract_image_features(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features

def concat(hash,name,feature):
    try:
        if hash[name]:
            print('presented')
    except:
        hash[name] = feature

def add(hash,name,image):
    feature = extract_image_features(Image.fromarray(image))
    concat(hash,name,feature)

def find_closest(hash, new_feature):
    
    similar = -float('inf')
    
    for name,feature in hash.items():
        if similar < torch.nn.functional.cosine_similarity(feature, new_feature).item():
            similar = feature
            close_name = name
    if close_name is None:
        raise ValueError("The features dictionary is empty.")
    
    return close_name, similar

def capture_image(cap):
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
    
    results = detect(img)
    for res in results:
        boxes = res.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if cls == 0 and conf > 0.6:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  
                crop = img[y1:y2, x1:x2]         
        cv2.imshow('img', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return None
    if crop is not None:
        name = input('Enter name:')
        return crop, name
    else:
        print("No valid object detected.")
        return None, None