import os
from flask import Flask, render_template
from flask_jwt_extended import JWTManager #type: ignore

from auth import auth_bp

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'Frontend', 'templates'),
            static_folder=os.path.join(BASE_DIR, 'Frontend', 'static'))
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', app.config['SECRET_KEY'])
app.config['APP_NAME'] = 'Nebula Lock'
app.config['APP_YEAR'] = '2025'

# Performance optimizations
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # Cache static files for 1 year
app.config['JSON_SORT_KEYS'] = False  # Don't sort JSON keys (faster)

JWTManager(app)
app.register_blueprint(auth_bp, url_prefix='/')


# Warmup DeepFace model on startup to avoid first-request delay
def warmup_models():
    """Pre-load DeepFace models to reduce first-request latency."""
    try:
        print("🔥 Warming up face recognition models...")
        import numpy as np
        from Utils.utils import gen_embeddings
        
        # Create a dummy 224x224 RGB image
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # This will download and cache the model
        gen_embeddings(dummy_img)
        print("✓ Models loaded and ready!")
    except Exception as e:
        print(f"⚠ Model warmup failed (will load on first use): {e}")


# Warmup in background on startup
import threading
threading.Thread(target=warmup_models, daemon=True).start()


@app.route('/')
def home():
    return render_template('auth_ui.html')


@app.route('/health')
def health_check():
    return {'status': 'ok', 'message': 'Server running'}


if __name__ == "__main__":
    app.run(debug=True)
