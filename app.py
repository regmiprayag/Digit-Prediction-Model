from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np 
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if 'image' in request.files:
        #Handle file upload 
        file = request.files['image']
        img = Image.open(file.stream)

    else:
        # Handle canvas drawing
        img_data = request.form['image'].split('.')[1]
        img = Image.open(io.BytesIO(base64.b64decode(img_data)))

    #preprocess the image
    img = img.convert('L').resize((28,28))
    img_array = np.array(img).reshape(1,28,28,1).astype('float32')/255

    # Make Prediction
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    return jsonify({'digit':int(digit),'confidence':float(prediction[0][digit])})

if __name__ == '__main__':
    app.run()
