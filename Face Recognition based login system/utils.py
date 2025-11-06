import cv2
from deepface import DeepFace  # type: ignore
import base64 
import numpy as np
from numpy.linalg import norm
import os 
from dotenv import load_dotenv
import json

load_dotenv()

"""Constants"""
USERS_FILE=os.getenv("USERS_FILE")
EMBEDDING_DIR=os.getenv("EMBEDDING_DIRECTORE")
jwt_secret_key=os.getenv("JWT_SECRET_KEY")



"""Validation Functions"""

def validate_username(username):
    if not username:
        return False,"Username is required"
    if len(username)<5 or len(username)>20:
        return False,"The length of username should be between 5 and 20"
    if not username.isidentifier():
        return False,"Usernames can only contain letters, numbers and underscores only"
    
    return True,"Valid Username"

def validate_password(password):
    
    special_chars=r"!@#$%^&*()-_+=<>?/{}~|"
    if not password:
        return False,"Password is required"
    if len(password)<8:
        return False,"Passwords must be atleast 8 characters in length"
    if not any(c.islower() for c in password):
        return False,"Password should contain atleast 1 lowercase character"
    if not any(c.isupper() for c in password):
        return False,"Password should contain atleast 1 uppercase character"
    if not any(c.isdigit() for c in password):
        return False,"Password should contain atleast 1 digit"
    if not any(c in special_chars for c in password):
        return False,"Password must contain atleast 1 special character"
    return True,"Password Accepted"


"""Image Recognition helpers"""

def decode_image(face_b64):
    data=base64.b64decode(face_b64)
    arr=np.frombuffer(data,np.uint8)
    return cv2.imdecode(arr,cv2.IMREAD_COLOR)


def get_embeddings(image):
    try:
        out=DeepFace.represent(img_path=image,model_name="",enforce_detection=True)
        emb=np.array(out[0]['embedding'],dtype=np.float32)
        emb /=norm(emb)
        return emb 
    except Exception as e:
        print(e)
        return None

def compare_embeddings(emb1,emb2,threshold=0.25):
    cos_sim=np.dot(emb1,emb2)/(norm(emb1)+norm(emb2))
    return cos_sim>(1-threshold)

"""User Management"""

def load_users():
    if os.path.exists(USERS_FILE): #type: ignore
        with open(USERS_FILE,'r') as f: #type: ignore
            return json.load(f)
    return {} 

def save_users(users):
    if os.path.exists(USERS_FILE): #type: ignore
        with open(USERS_FILE,'a') as f: #type: ignore   
            json.dump(users,f,indent=4)
            