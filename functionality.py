import torch
from source.clip.clip import clip
from PIL import Image
import cv2
from ultralytics import YOLO
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = clip.load("./weights/ViT-L-14.pt", device=device)
detect = YOLO('./weights/yolov8l.pt')
detect.to(device)
features_hash = {}

def extract_image_features(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features

def add(name_input,crop):
    global features_hash
    feature = extract_image_features(Image.fromarray(crop))
    if name_input not in features_hash:
        features_hash[name_input] = feature
        return f"Added {name_input} to the database."

def remove(name_input):
    global features_hash
    if name_input in features_hash:
        del features_hash[name_input]
        return f'Removed {name_input} from the database'
    return f'{name_input} not found'

def clear():
    global features_hash
    features_hash = {}
    return 'Cleared all features'
def find_closest(features_hash, new_feature):
    if not features_hash:
        raise ValueError("The features dictionary is empty.")
    
    close_name = max(features_hash, key=lambda name: torch.nn.functional.cosine_similarity(features_hash[name], new_feature).item())
    similar = torch.nn.functional.cosine_similarity(features_hash[close_name], new_feature).item()
    
    return close_name, similar

def process_frame(frame,name_input):
    global features_hash
    results = detect(frame, stream=True)
    message = "Face not detected"
    for res in results:
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            
            if cls == 0 and conf > 0.7:
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    feature = extract_image_features(Image.fromarray(crop))
                    if features_hash:
                        name, similarity = find_closest(features_hash,feature)
                        if similarity > 0.7:
                            message = f"Hello {name}"
                        else:
                            if name_input:
                                message = add(name_input, crop)
                            else:
                                message = "No faces in the database. Please enter a name to add this face."
                    else:
                        if name_input:
                            message = add(name_input, crop)
                        else:
                            message = "No faces in the database. Please enter a name to add this face."

    return frame, message

def capture_and_process_frames(frame_queue,name_input):
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = detect(frame, stream=True)
        message = "Face not detected"
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                
                if cls == 0 and conf > 0.7:
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        feature = extract_image_features(Image.fromarray(crop))
                        if features_hash:
                            name, similarity = find_closest(features_hash, feature)
                            if similarity > 0.7:
                                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                                message = f"Hello {name}"
                            else:
                                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                if name_input:
                                    message = add(features_hash, name_input, crop)
                                else:
                                    message = "No faces in the database. Please enter a name to add this face."
                        else:
                            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            if name_input:
                                message = add(features_hash, name_input, crop)
                            else:
                                message = "No faces in the database. Please enter a name to add this face."

        
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put((frame, message))
