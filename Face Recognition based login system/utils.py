import base64
from deepface import DeepFace #type: ignore 
import numpy as np
import cv2
import os
from numpy.linalg import norm
from dotenv import load_dotenv
import json

load_dotenv()


"""Constants"""
user_file=os.getenv("USER_FILE")
embedding_dir=os.getenv("EMBEDDING_DIR")
jwt_secret_key=os.getenv("JWT_SECRET_KEY")


"""User managemetn"""

def load_users():
    if os.path.exists(user_file): #type: ignore 
        with open(user_file,'r') as f: #type: ignore
            return json.loads(f) 
    
    return {}

def save_users(users):
    with open(user_file,'w') as f: #type: ignore
        json.dump(users,f,indent=4)
        
            
    
"""Image Embedding generation"""


def decode_base64_image(image):
    data=base64.b64decode(image)
    arr=np.frombuffer(data,np.uint8)
    return cv2.imdecode(arr,cv2.IMREAD_COLOR) 

def get_embeddings(image):
    try:
        out=DeepFace.represent(img_path=image,model_name="Facenet",enforce_detection=True)
        emb=np.array(out[0]['embedding'],dtype=np.float32)
        emb /=norm(emb)
        return emb
    except Exception as e:
        print(e)
        return None
    
    
def compare_embeddings(emb1,emb2,threshold=0.25):
    cos_sim=np.dot(emb1,emb2)/(norm(emb1)* norm(emb2))
    return cos_sim > (1-threshold)
