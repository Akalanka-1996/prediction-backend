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
import numpy as np
import os
from PIL import Image
import io
import base64
import cv2


imageModel = tf.keras.models.load_model('final.h5')

app = Flask(__name__)
CORS(app)


def imagePrediction(img_data, target):
    npimg = np.fromstring(img_data, np.uint8)
    npimg = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.imwrite('prediction.jpg', npimg)
    
    classes = ['gender_male', 'isMarried_True', 'race_Muslim', 'race_Sinhala', 'race_Tamil', 'religion_Buddhism', 'religion_Catholic','religion_Hindu', 'religion_Islam']
    
    img = keras_image.load_img('prediction.jpg', target_size=target)
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0
    
    proba = imageModel.predict(np.array([img_array]))
    top_3 = np.argsort(proba[0])[:-4:-1]
    top_predictions = []
    for i in range(3):
        prediction = {
            "class": classes[top_3[i]],
            "probability": "{:.3}".format(proba[0][top_3[i]])
        }
        top_predictions.append(prediction)
        print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))

    data = {
            "title": "Prediction Result",
            "predictions": top_predictions
    }
    response = flask.jsonify(data)

    response.headers.add('Access-Control-Allow-Origin', '*')
    print(data)

    return response

@app.route('/')
def hello_world():
    return 'Hello, Flask World!'

@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            if flask.request.files.get("image"):
                result = ''
                pred_image = request.files['image'].read()
                result = imagePrediction(pred_image, target=(224, 224))
                return (result)

            return ({"status": false}) 
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "error_message": str(e)})

if __name__ == '__main__':
    app.run(port=3001)