from flask import Flask
import flask
from flask import jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, Flask World!'

if __name__ == '__main__':
    app.run(port=3001)