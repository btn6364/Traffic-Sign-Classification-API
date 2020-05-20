import os
from PIL import Image
import numpy
from keras.models import load_model

#########################################################
# SETTING FOR MEMORY ALLOCATION
#########################################################

import tensorflow as tf
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

########################################################
# PATHS
########################################################

BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "output/traffic_sign_classifier.h5") 
IMAGE_PATH = os.path.join(BASE_PATH, "testing_images")
SIGNS_PATH = os.path.join(BASE_PATH, "../data/Signs.csv")


########################################################
# LOAD MODEL
########################################################
model = load_model(MODEL_PATH)


########################################################
# HELPER FUNCTIONS
########################################################

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

def classify_all_testing_images(classes):
    paths = os.listdir(IMAGE_PATH)
    for path in paths:
        path = os.path.join(IMAGE_PATH, path)
        pred = classify(path, classes)
        print(f"[PREDICTED RESULT for {path}]: {pred}")

def main():
    classes = load_classes()
    classify_all_testing_images(classes)

if __name__ == "__main__":
    main()
