from PIL import Image
import tensorflow as tf
import cv2
import numpy as np

from flask import Flask, render_template,request
app = Flask(__name__)

def start_prediction(fileName):
    model_path = "model_accuracy_0.9407.h5"
    loaded_model = tf.keras.models.load_model(model_path)
    image = cv2.imread(fileName)

    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((224, 224))
    expand_input = np.expand_dims(resize_image, axis=0)
    input_data = np.array(expand_input)
    input_data = input_data / 255

    pred = loaded_model.predict(input_data)
    if pred >= 0.5:
        print("Breast Cancer: Yes")
        return render_template("detect.html", result=True)
    else:
        print("Breast Cancer: No")
        return render_template("detect.html", result=False)

@app.route('/')
def root():
    return render_template("index.html")

@app.route('/detect')
def detect():
    return render_template("detect.html")

@app.route('/predict',methods=["POST"])
def predict():
    if request.method == "POST":
        print("POST REQUEST")
        histoImage = request.files["histoImage"]
        fileName = "static/images/"+histoImage.filename
        histoImage.save(fileName)

        # start_prediction("images/"+histoImage.filename)
        model_path = "model_accuracy_0.9407.h5"
        loaded_model = tf.keras.models.load_model(model_path)
        image = cv2.imread(fileName)

        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((224, 224))
        expand_input = np.expand_dims(resize_image, axis=0)
        input_data = np.array(expand_input)
        input_data = input_data / 255


        pred = loaded_model.predict(input_data)
        if pred >= 0.5:
            print("Breast Cancer: Yes")
            return render_template("detect.html", result="Breast Cancer is Detected", img_path="http://localhost:3000/"+fileName)
        else:
            print("Breast Cancer: No")
            return render_template("detect.html", result="Breast Cancer is Not Detected", img_path="http://localhost:3000/"+fileName)



if __name__ == "__main__":
    app.run(port=3000)

