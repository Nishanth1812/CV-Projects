"""Production-ready Flask application launcher with optimizations."""
import os
from app import app

if __name__ == "__main__":
    # Try to enable compression if available
    try:
        from flask_compress import Compress
        Compress(app)
        print("✓ Compression enabled")
    except ImportError:
        print("ℹ Install flask-compress for better performance: pip install flask-compress")
    
    # Production server
    port = int(os.getenv('PORT', 5000))
    
    print(f"\n{'='*60}")
    print("NEBULA LOCK - Production Mode")
    print(f"{'='*60}")
    print(f"Server: http://localhost:{port}")
    print(f"Environment: {'Production' if not app.debug else 'Development'}")
    print(f"{'='*60}\n")
    
    # Use waitress on Windows, gunicorn on Linux/Mac
    try:
        from waitress import serve
        print("✓ Using Waitress WSGI server (optimized for Windows)")
        serve(app, host='0.0.0.0', port=port, threads=4)
    except ImportError:
        try:
            # Fallback: try gunicorn command
            os.system(f"gunicorn -w 4 -b 0.0.0.0:{port} app:app")
        except:
            print("⚠ For better performance, install: pip install waitress")
            print("  Running with Flask dev server...")
            app.run(host='0.0.0.0', port=port, debug=False)
