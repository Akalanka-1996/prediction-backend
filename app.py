from flask import Flask, render_template, request
import flask
from flask import jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image as keras_image

# from flask_mongoengine import MongoEngine
# from flask_mongoengine import MongoEngine
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from flask_pymongo import PyMongo
import numpy as np
import os
from PIL import Image
import io
import base64
import cv2
import random
import string
import datetime


imageModel = tf.keras.models.load_model("final.h5")

app = Flask(__name__)
CORS(app)
app.config["MONGO_URI"] = "mongodb://localhost:27017/prediction"
app.config["SECRET_KEY"] = "wgTdWV9jtJ7nqQJk" 
mongo = PyMongo(app)


def generate_random_user_id():
    random_suffix = "".join(random.choices(string.digits, k=6))
    user_id = f"USER{random_suffix}"
    return user_id


def create_jwt(user_id):
    payload = {
        "user_id": user_id,
        "exp": datetime.datetime.utcnow()
        + datetime.timedelta(days=1),  # Token expiration time
    }
    token = jwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")
    return token


def imagePrediction(img_data, target):
    npimg = np.fromstring(img_data, np.uint8)
    npimg = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.imwrite("prediction.jpg", npimg)

    classes = [
        "gender_male",
        "isMarried_True",
        "race_Muslim",
        "race_Sinhala",
        "race_Tamil",
        "religion_Buddhism",
        "religion_Catholic",
        "religion_Hindu",
        "religion_Islam",
    ]

    img = keras_image.load_img("prediction.jpg", target_size=target)
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0

    proba = imageModel.predict(np.array([img_array]))
    top_3 = np.argsort(proba[0])[:-4:-1]
    top_predictions = []
    for i in range(3):
        prediction = {
            "class": classes[top_3[i]],
            "probability": "{:.3}".format(proba[0][top_3[i]]),
        }
        top_predictions.append(prediction)
        print("{}".format(classes[top_3[i]]) + " ({:.3})".format(proba[0][top_3[i]]))

    data = {"title": "Prediction Result", "predictions": top_predictions}
    response = flask.jsonify(data)

    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


@app.route("/")
def hello_world():
    return "Hello, Flask World!"


@app.route("/prediction", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "POST":
            if flask.request.files.get("image"):
                result = ""
                pred_image = request.files["image"].read()
                result = imagePrediction(pred_image, target=(224, 224))
                return result

            return {"status": false}
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "error_message": str(e)})


@app.route("/register", methods=["POST"])
def register():
    data = request.json

    if not data or "username" not in data or "password" not in data:
        return jsonify({"message": "Missing username or password"}), 400

    username = data["username"]
    password = data["password"]
    user_type = data["user_type"]

    if user_type not in ["designer", "vendor"]:
        return jsonify({"message": "Invalid user type"}), 400

    existing_user = mongo.db.users.find_one({"username": username})
    if existing_user:
        return jsonify({"message": "Username already exists"}), 400

    user_id = generate_random_user_id()
    hashed_password = generate_password_hash(password, method="sha256")

    new_user = {
        "user_id": user_id,
        "username": username,
        "password": hashed_password,
        "user_type": user_type,
    }
    mongo.db.users.insert_one(new_user)
    jwt = create_jwt(user_id)

    return (
        jsonify(
            {
                "message": "User registered successfully",
                "token": jwt,
                "user": {
                    "user_id": user_id,
                    "username": username,
                    "user_type": user_type,
                },
            }
        ),
        201,
    )


if __name__ == "__main__":
    app.run(port=3001)
