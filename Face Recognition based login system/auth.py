from flask import Blueprint,request,jsonify,flash,session,redirect,url_for,render_template
from flask_jwt_extended import create_access_token #type: ignore
from werkzeug.security import generate_password_hash,check_password_hash
from dotenv import load_dotenv
import os


load_dotenv()


auth_bp=Blueprint("Auth",__name__)

"""Creating the register route"""
@auth_bp.route("/register",methods=['GET','POST'])
def register():
    data=request.get_json()
    
    username=data.get("userna")
    return ""
