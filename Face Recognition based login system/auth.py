from flask import Blueprint, request, flash, session, render_template
from flask_jwt_extended import create_access_token #type: ignore
from werkzeug.security import generate_password_hash, check_password_hash
from Utils.utils import gen_embeddings, compare_embeddings, decode_image
from Utils.db_utils import save_user, user_exists, save_embedding, load_embedding, load_users

auth_bp = Blueprint("Auth", __name__)


@auth_bp.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    confirm_pass = data.get("confirm_password", "")
    email = data.get("email", "").strip()
    face = data.get("face_image")

    if not username:
        flash("Username is required.", 'error')
        return {'status': 'error', 'message': "Username is required."}, 400
    if not email:
        flash("Email is required.", 'error')
        return {'status': 'error', 'message': "Email is required."}, 400
    if not password:
        flash("Password is required.", 'error')
        return {'status': 'error', 'message': "Password is required."}, 400
    if password != confirm_pass:
        flash("Passwords do not match.", 'error')
        return {'status': 'error', 'message': "Passwords do not match."}, 400
    if not face:
        flash("Face snapshot missing.", 'error')
        return {'status': 'error', 'message': "Face snapshot missing."}, 400

    if user_exists(username):
        flash('User already registered.', 'error')
        return {'status': 'error', 'message': 'User already registered.'}, 409

    import time
    start = time.time()
    
    img = decode_image(face)
    if img is None:
        flash('Unable to decode face snapshot.', 'error')
        return {'status': 'error', 'message': 'Unable to decode face snapshot.'}, 400
    
    print(f"Image decode time: {time.time() - start:.2f}s")
    decode_time = time.time()

    embeddings = gen_embeddings(img)
    if embeddings is None:
        flash('Unable to process facial embedding.', 'error')
        return {'status': 'error', 'message': 'Unable to process facial embedding.'}, 400
    
    print(f"Embedding generation time: {time.time() - decode_time:.2f}s")
    embed_time = time.time()

    try:
        save_user(username, email, generate_password_hash(password))
        save_embedding(username, embeddings)
        print(f"Database save time: {time.time() - embed_time:.2f}s")
        print(f"Total registration time: {time.time() - start:.2f}s")
        flash('User registered successfully.', 'success')
        return {'status': 'success', 'message': 'User registered successfully.'}
    except Exception as e:
        flash(f'Registration failed: {str(e)}', 'error')
        return {'status': 'error', 'message': f'Registration failed: {str(e)}'}, 500

@auth_bp.route("/login", methods=['GET'])
def login_page():
    return render_template('login.html')


@auth_bp.route("/dashboard", methods=['GET'])
def dashboard():
    username = session.get('username')
    token = session.get('access token', '')
    token_preview = f"{token[:18]}…{token[-6:]}" if len(token) > 32 else token if token else None
    face_ready = load_embedding(username) is not None if username else False
    return render_template('dashboard.html', username=username, face_ready=face_ready, token_preview=token_preview)


@auth_bp.route("/login_face", methods=['POST'])
def login_face():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    face = data.get("face_image")
    attempt = int(data.get('attempt', 1))

    if not username:
        flash("Username is required.", 'error')
        return {'status': 'error', 'message': "Username is required."}, 400
    if not face:
        flash("Face snapshot missing.", 'error')
        return {'status': 'error', 'message': "Face snapshot missing."}, 400

    if not user_exists(username):
        flash('User not found.', 'error')
        return {'status': 'error', 'message': 'User not found.'}, 404

    stored_emb = load_embedding(username)
    if stored_emb is None:
        flash('No stored embedding found. Please re-register.', 'error')
        return {'status': 'error', 'message': 'No stored embedding found.'}, 404

    img = decode_image(face)
    if img is None:
        msg = 'Unable to read face.' if attempt < 3 else 'Unable to read facial data. Use credential login.'
        flash(msg, 'warning' if attempt < 3 else 'error')
        status = 'retry' if attempt < 3 else 'error'
        return {'status': status, 'message': msg, 'attempt': attempt + 1}, 401

    embedding = gen_embeddings(img)
    if embedding is None or not compare_embeddings(embedding, stored_emb, threshold=0.75):
        msg = f'Face not recognized. {3 - attempt} attempt(s) left.' if attempt < 3 else 'Face authentication failed. Use credential login.'
        flash(msg, 'warning' if attempt < 3 else 'error')
        status = 'retry' if attempt < 3 else 'error'
        return {'status': status, 'message': msg, 'attempt': attempt + 1}, 401

    session['username'] = username
    session['access token'] = create_access_token(identity=username)
    flash('Authentication successful.', 'success')
    return {'status': 'success', 'message': 'Authentication successful.'}


@auth_bp.route("/login_cred", methods=['POST'])
def login_cred():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username:
        flash("Username is required.", 'error')
        return {'status': 'error', 'message': "Username is required."}, 400
    if not password:
        flash("Password is required.", 'error')
        return {'status': 'error', 'message': "Password is required."}, 400

    user = (load_users() or {}).get(username)
    if not user:
        flash('User not found.', 'error')
        return {'status': 'error', 'message': 'User not found.'}, 404

    if not user.get('password') or not check_password_hash(user['password'], password):
        flash('Incorrect username or password.', 'error')
        return {'status': 'error', 'message': 'Incorrect username or password.'}, 401

    session['username'] = username
    session['access token'] = create_access_token(identity=username)
    flash('Login successful.', 'success')
    return {'status': 'success', 'message': 'Login successful.'}


    
    