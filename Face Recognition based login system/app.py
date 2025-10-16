from flask import Flask,jsonify
from auth import auth_bp

app=Flask(__name__)

app.register_blueprint(auth_bp,url_prefix='/')

@app.route('/')
def health_check():
    return jsonify({"message":"Server running well"})

if __name__=="__main__":
    app.run(debug=True)
    
