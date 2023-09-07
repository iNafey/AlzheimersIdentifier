import base64
import numpy as np
import io
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.models import Sequential, load_model
from keras.utils.image_utils import img_to_array
from flask_cors import CORS

# The code for the model loading, input preprocessing and POST request was adapted from a website (link below)
#https://www.analyticsvidhya.com/blog/2022/01/develop-and-deploy-image-classifier-using-flask-part-2/?utm_source=related_WP&utm_medium=https://www.analyticsvidhya.com/blog/2022/01/develop-and-deploy-image-classifier-using-flask-part-1/


app = Flask(__name__)
CORS(app)

def get_model():
    global model
    model = tf.keras.models.load_model('models/custom_finetuned_model')
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image =  image.convert("RGB")

    image = image.resize(target_size)
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image, axis=0)
    print(image.shape)

    return image

print(" * Loading Keras CNN model...")
get_model()

@app.route('/predict', methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_img = preprocess_image(image, target_size=(224, 224))

    prediction = model.predict(processed_img)
    pred_list = prediction[0].tolist()

    print(pred_list)

    mild_prob =  str(100*round(pred_list[0], 4))
    mod_prob = str(100*round(pred_list[1], 4))
    cn_prob = str(100*round(pred_list[2], 4))
    vmild_prob = str(100*round(pred_list[3], 4))

    response = {
        'prediction': {
            'mild': mild_prob,
            'moderate': mod_prob,
            'neutral': cn_prob,
            'vmild': vmild_prob
        }
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host = '127.0.0.1' ,port = 5000)
