from flask import Blueprint,request,jsonify,flash,session,redirect,url_for,render_template
from flask_jwt_extended import create_access_token #type: ignore
from werkzeug.security import generate_password_hash,check_password_hash
from dotenv import load_dotenv
from utils import get_embeddings,compare_embeddings,decode_base64_image,load_users,save_users
import numpy as np
import os


load_dotenv()


auth_bp=Blueprint("Auth",__name__)

"""Creating the register route"""
@auth_bp.route("/register",methods=['GET','POST'])
def register():
    data=request.get_json()
    
    username=data.get("username")
    password=data.get("password")
    confirm_pass=data.get("confirm_password")
    face=data.get("face_image")
    
    if not username:
        flash("Username is required")
    if not password:
        flash("Password is required")
    
    users=load_users() 
    if not users:
        flash("Unable to load the users data")
    
    if username in users:
        flash("User is already registered")
        
    
    img=decode_base64_image(face)
    if img is None:
        flash("Face couldnt be detected")
        
    embeddings=get_embeddings(img)
    if embeddings is None:
        flash("Unable to process image")
        
    embedding_path=os.path.join(os.getenv("EMBEDDINGS_DIR"), f"{username}.npy") #type: ignore
    
    np.save(embedding_path, embeddings) #type: ignore
    
    if password!=confirm_pass:
        flash("The passwords do not match, try again")
    
    hashed_pass=generate_password_hash(password=password)
    
    users[username] = {'password': hashed_pass}
    save_users(users)
    
    return jsonify({'status': 'success', 'message': 'Face embedding generated and user registered successfully'})

@auth_bp.route("/login_face",methods=['POST'])
def login_face():
    data=request.get_json()
    username=data.get("usernmae")
    face=data.get("face_image")
    attempt = data.get('attempt', 1)
    
    
    if not username:
        flash("Username is requried")
        
    if not face:
        flash("No face has been detected")
    
    users=load_users()
    if not users:
        flash("Couldn't process user details")
    
    if username not in users:
        flash("The user does not exist")
    
    img=decode_base64_image(image=face)
    
    if not img:
        flash("Couldn't process face")
    
    embedding=get_embeddings(image=img)
    
    if embedding is None:
        if attempt<3:
            return jsonify({
                'status': 'retry',
                'message': f'Face not recognized. {3-attempt} attempt(s) left.',
                'attempt': attempt + 1
            }), 401
    
    embedding_path = os.path.join(os.getenv("EMBEDDING_DIR"), f"{username}.npy") #type: ignore 
    
    stored_emb=np.load(embedding_path)
    
    if not compare_embeddings(emb1=embedding,emb2=stored_emb):
        flash("Sorry face does not match")
        
    token=create_access_token(identity=username)
    session['access token']=token 
    return  jsonify({'status': 'success'})

@auth_bp.route("/login_cred",methods=['POST'])
def login_cred():
    data=request.get_json()
    username=data.get("username")
    password=data.get("password")
    
    if not username:
        flash("Username is missing")
    
    if not password:
        flash("Password is required")
        
    users=load_users()
    
    if not users:
        flash("Unable to fetch user data")
    
    hashed_pass=users[username]['password']
    
    if not check_password_hash(hashed_pass,password=password):
        flash("Incorrect password")
    
    token=create_access_token(identity=username)
    
    return jsonify({'status': 'success'})


    
    