from flask import Flask, request, jsonify, render_template, redirect, session
from PIL import Image
import numpy
import os
from keras.models import load_model

########################################################
# PATHS
########################################################

BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "../model/output/traffic_sign_classifier.h5") 
SIGNS_PATH = os.path.join(BASE_PATH, "../data/Signs.csv")

# ####################################################
# # SETTING FOR GROWTH MEMORY ALLOCATION
# ####################################################
# import tensorflow as tf
# gpus= tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


# ##################################
# # LOAD THE MODEL
# ##################################
# model = load_model(MODEL_PATH)



##################################
# HELPER FUNCTIONS
##################################

def load_classes():
    classes = {}
    lines = open(SIGNS_PATH).read().strip().split("\n")[1:]
    for line in lines:
        number, description = line.strip().split(",")
        classes[int(number) + 1] = description
    return classes


def classify(file_path, classes):
    try:
        image = Image.open(file_path)
        image = image.resize((30, 30))
        image = numpy.expand_dims(image, axis=0)
        image = numpy.array(image)
        pred = model.predict_classes([image])[0]
        sign = classes[pred + 1]
        return sign
    except:
        print("[ERROR]: Error when loading images")


##################################
# CREATE AND APPLICATION
##################################
app = Flask(__name__)
app.config["DEBUG"] = True
app.config["IMAGE_UPLOADS"] = os.path.join(BASE_PATH, "uploads")
app.secret_key = b'@\x93,Cby\xae\xf8\x83\xe3D\x06\x9e\x98\x8f\xa6'

##################################
# ROUTE HANDLERS
##################################

@app.route("/", methods=["GET"])
@app.route("/home", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/history", methods=["GET"])
def history():
    return render_template("history.html")


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
             #save the image to session
            image = request.files["image"]
            save_path = os.path.join(app.config["IMAGE_UPLOADS"], image.filename) 
            image.save(save_path)
            session["image"] = save_path
            return redirect("/upload-image/predict")
    return render_template("upload.html")


@app.route("/upload-image/predict", methods=["GET"])
def predict_image():
    if "image" in session:
        print(f"Image path is : {session['image']}")
    else:
        print("Cannot found image!")
    return render_template("predict.html")


@app.errorhandler(404)
def page_not_found(err):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run()
