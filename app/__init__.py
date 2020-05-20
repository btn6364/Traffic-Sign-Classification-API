from flask import Flask, request, jsonify
from PIL import Image
import numpy
import os
from keras.models import load_model

##################################
# CONSTANTS 
##################################

BASE_PATH = os.path.abspath(os.path.dirname(__file__))


####################################################
# SETTING FOR GROWTH MEMORY ALLOCATION
####################################################
import tensorflow as tf
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


##################################
# CREATE AND APPLICATION
##################################
app = Flask(__name__)
app.config["DEBUG"] = True


##################################
# LOAD THE MODEL
##################################
path = os.path.join(BASE_PATH, "../model/traffic_sign_classifier.h5")
model = load_model(path)


##################################
# ROUTE HANDLERS
##################################

@app.route("/", methods=["GET"])
def home():
    return "Welcome to home page!"

@app.errorhandler(404)
def page_not_found(err):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


if __name__ == "__main__":
    app.run()
