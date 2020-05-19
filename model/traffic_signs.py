###############################################
# RUN THIS CODE FOR CREATING AND SAVING MODELS
###############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import accuracy_score

BATCH_SIZE = 64
EPOCHS = 15
BASE_PATH = os.path.abspath(os.path.dirname(__file__))

####################################################
# SETTING FOR GROWTH MEMORY ALLOCATION
####################################################
import tensorflow as tf
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


"""
Load images dataset and their labels
"""
def load_images_labels():
    data = []
    labels = []
    classes = 43
    for i in range(classes):
        path = os.path.join(BASE_PATH, "../data/train", str(i))
        images = os.listdir(path)
        for img in images:
            try:
                image = Image.open(path + "/" + img)
                image = image.resize((30, 30))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image...")
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

"""
Split the test and training data
"""
def split_test_train():
    data, labels = load_images_labels()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)
    return X_train, X_test, y_train, y_test

"""
Build a model with layers
"""
def build_model(X_train):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    return model

"""
Plot the stat graph of the training. 
"""
def stat_graph(history):
    plt.figure(0)
    plt.plot(history.history["accuracy"], label="training accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()

    plt.figure(1)
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()

"""
Test the model
"""
def test_model(X_test, y_test, model):
    test_path = os.path.join(BASE_PATH, "../data/Test.csv")
    y_test = pd.read_csv(test_path)
    labels = y_test["ClassId"].values
    imgs = y_test["Path"].values
    data = []
    for img in imgs:
        image_path = os.path.join(BASE_PATH, "../data/", img)
        image = Image.open(image_path)
        image = image.resize((30, 30))
        data.append(np.array(image))
    X_test = np.array(data)
    pred = model.predict_classes(X_test)
    #Accuracy with the test data
    accuracy_score(labels, pred)



def main():
    X_train, X_test, y_train, y_test = split_test_train()
    
    #get the model
    model = build_model(X_train)

    #compile and train the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(X_train, y_train, batch_size=64, epochs=EPOCHS, validation_data=(X_test, y_test))

    #draw stat graph
    stat_graph(history)

    #test the model
    test_model(X_test, y_test, model)

    #save the model
    model.save("traffic_sign_classifier.h5")
    
if __name__ == "__main__":
    main()