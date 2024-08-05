import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
from PIL import Image
import numpy as np
from ultralytics import YOLO
from clip.clip import clip
import faiss
import math
import os

animal_classes = [
    "dog", "cat", "bird", "horse", "deer", "frog", "fish", "turtle", 
    "elephant", "tiger", "lion", "monkey", "bear", "rabbit", "mouse", 
    "cow", "sheep", "goat", "panda", "kangaroo", "koala", "zebra", 
    "giraffe", "penguin", "dolphin", "shark", "whale", "crocodile", 
    "alligator", "snake", "lizard", "chameleon", "iguana", "octopus", 
    "crab", "lobster", "spider", "ant", "bee", "butterfly", "bat", 
    "hedgehog", "otter", "raccoon", "squirrel", "owl", "eagle", 
    "parrot", "flamingo", "peacock", "swan", "goose", "duck", 
    "chicken", "rooster", "turkey", "pigeon", "dove", "seal", 
    "walrus", "manatee", "hippopotamus", "rhinoceros", "buffalo", 
    "bison", "camel", "donkey", "mule", "chimpanzee", "gorilla", 
    "orangutan", "lemur", "meerkat", "mongoose", "porcupine", 
    "skunk", "beaver", "armadillo", "sloth", "tapir", "platypus", 
    "wombat", "kangaroo rat", "badger", "mole", "weasel", "ferret", 
    "lynx", "bobcat", "cheetah", "panther", "jaguar", "cougar", 
    "snow leopard", "gazelle", "antelope", "moose", "reindeer", 
    "elk", "caribou", "narwhal", "beluga", "orca", "seal", 
    "walrus", "manta ray", "jellyfish", "starfish", "seahorse"
]

human_classes = [
    "pretty girl", "handsome boy", "old man", "young woman", "smiling child",
    "angry man", "happy woman", "crying baby", "laughing teenager", "serious adult",
    "elderly person", "athletic man", "strong woman", "cute baby", "stylish girl",
    "confident boy", "tired man", "excited woman", "nervous child", "relaxed person",
    "worried mother", "proud father", "brave soldier", "smart student", "curious kid",
    "gentle grandma", "wise grandpa", "kind teacher", "friendly neighbor", "bossy manager",
    "cheerful nurse", "angry boss", "calm doctor", "diligent worker", "funny comedian",
    "quiet librarian", "adventurous explorer", "loyal friend", "creative artist",
    "talented musician", "grumpy old man", "energetic young girl", "playful boy",
    "serious scientist", "dedicated athlete", "passionate dancer", "thoughtful writer",
    "caring nurse", "strict teacher", "motivated student", "loving parent",
    "supportive sibling", "humble hero", "competitive player", "determined leader"
]

path = r'D:\Data\AnimalsDataSample'
images = os.listdir(path)
image_paths = [os.path.join(path, image) for image in images]


class Processor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.preprocess = None, None
        self.detect = None
        self.features_hash: Dict[str, torch.Tensor] = {}
        self.name_to_index: Dict[str, int] = {}
        self.index_to_name: Dict[int, str] = {}
        self.animal_classes = self.extract_text_features(animal_classes)
        self.human_classes = self.extract_text_features(human_classes)
        
    @lru_cache(maxsize=None)
    def load_models(self):
        if self.model is None:
            self.model, self.preprocess = clip.load("./weights/ViT-L-14.pt", device=self.device)
        if self.detect is None:
            self.detect = YOLO('./weights/yolov8l.pt')
            self.detect.to(self.device)
            
    @torch.no_grad()
    def extract_text_features(self, classes: List[str]) -> torch.Tensor:
        self.load_models()
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(self.device)
        classes_encoded = self.model.encode_text(text_inputs)
        classes_encoded /= classes_encoded.norm(dim=-1, keepdim=True)
        return classes_encoded

    def process_images(self, image_paths: List[str], batch_size: int = 16):
        self.load_models()
        def batch_generator():
            for i in range(0, len(image_paths), batch_size):
                yield image_paths[i:i+batch_size]

        image_features_list = []
        for batch_paths in batch_generator():
            try:
                batch_tensors = torch.stack([self.preprocess(Image.open(img)) for img in batch_paths]).to(self.device)
                with torch.no_grad():
                    batch_features = self.model.encode_image(batch_tensors)
                image_features_list.append(batch_features)
            except Exception as e:
                print(f"Error processing batch: {e}")

        image_features = torch.cat(image_features_list)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    @torch.no_grad()
    def extract_image_features(self, image: Image.Image) -> torch.Tensor:
        self.load_models()
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def add(self, name_input: str, crop: np.ndarray) -> str:
        feature = self.extract_image_features(Image.fromarray(crop))
        self.features_hash[name_input] = feature
        return f"Added {name_input} to the database."

    def remove(self, name_input: str) -> str:
        if name_input in self.features_hash:
            del self.features_hash[name_input]
            return f'Removed {name_input} from the database'
        return f'{name_input} not found'

    def clear(self) -> str:
        self.features_hash.clear()
        return 'Cleared all features'

    def find_closest(self, new_feature: torch.Tensor) -> Tuple[str, float]:
        if not self.features_hash:
            raise ValueError("The features dictionary is empty.")

        similarities = []
        for name, feature in self.features_hash.items():
            similarity = torch.nn.functional.cosine_similarity(new_feature, feature)
            similarities.append((name, similarity.item()))
        
        if similarities:
            close_name, max_similarity = max(similarities, key=lambda x: x[1])
            return close_name, max_similarity
        else:
            return "", 0.0
    
class Feature(Processor):
    def __init__(self):
        super().__init__()
        self.image_features = self.process_images(image_paths)
        
    def process_frame(self, frame: np.ndarray, name_input: Optional[str] = None) -> Tuple[np.ndarray,str]:
        self.load_models()
        results = self.detect(frame, stream=True)
        message = "Face not detected"
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                
                if cls == 0 and conf > 0.7:
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        feature = self.extract_image_features(Image.fromarray(crop))
                        if self.features_hash:
                            name, similarity = self.find_closest(feature)
                            if similarity > 0.7:
                                message = f"Hello {name}"
                            elif name_input:
                                message = self.add(name_input, crop)
                            else:
                                message = "No matching faces. Please enter a name to add this face."
                        elif name_input:
                            message = self.add(name_input, crop)
                        else:
                            message = "No faces in the database. Please enter a name to add this face."
        return crop, message

    def look_like(self, image: np.ndarray) -> str:
        self.load_models()
        image = Image.fromarray(image)
        image_features = self.extract_image_features(image)
        similarity = (100.0 * image_features @ self.human_classes.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        return f"You look like a {human_classes[indices[0]]:>16s}"

    def animal(self, image: Optional[np.ndarray]) -> str:
        if image is None:
            return 'There is no image for inference'
        self.load_models()
        image = Image.fromarray(image)
        image_features = self.extract_image_features(image)
        similarity = (100.0 * image_features @ self.animal_classes.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        return f"This is a{animal_classes[indices[0]]:>16s} with the confidence of {100 * values[0].item():.2f}%"

    def animal_im(self, text: str) -> Image.Image:
        if text == '':
            return 'There is no text for inference'
        self.load_models()
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_feature = self.model.encode_text(text_input)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        similarity = (100.0 * text_feature @ self.image_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(15)
        top_indices = indices.cpu().numpy().flatten()
        top_values = values.cpu().numpy().flatten()

        probabilities = top_values / top_values.sum()

        match_index = np.random.choice(top_indices, p=probabilities)
        
        return Image.open(image_paths[match_index])

