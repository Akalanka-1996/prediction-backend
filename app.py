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
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import io


app = Flask(__name__)
CORS(app)


def imagePrediction(image, target):

    data = {
            "title": "Prediction Result"
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
                # result = ''
                # print('request', request.files)
                pred_image = request.files['image'].read()
                # image = tf.keras.utils.load_img(pred_image, target_size=(400,400,3))
                # input_arr = tf.keras.utils.img_to_array(image)
                # input_arr = np.array([input_arr])  # Convert single image to a batch.
                # print('done')
                # predictions = model.predict(input_arr)

                # return (result)
                
                img_stream = io.BytesIO(pred_image)

                # Load the image using tf.keras.preprocessing.image.load_img
                image = tf.keras.preprocessing.image.load_img(img_stream, target_size=(400, 400, 3))

                # Convert the image to an array
                input_arr = tf.keras.preprocessing.image.img_to_array(image)
                input_arr = np.array([input_arr])  # Convert single image to a batch.
                print('done')

            return ({"status": false}) 
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "error_message": str(e)})

if __name__ == '__main__':
    app.run(port=3001)