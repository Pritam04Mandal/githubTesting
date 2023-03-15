from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf
from PIL import Image
from rembg import remove
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
# from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from datetime import datetime
import sqlite3
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask import session
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'secret_key'


# Model saved with Keras model.save()
MODEL_PATH ='InceptionModel.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))


    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Tomato Bacterialn Spot"
    elif preds==1:
        preds="Tomato Early Blight"
    elif preds==2:
        preds="Tomato Late Blight"
    elif preds==3:
        preds="Tomato Leaf Mold"
    elif preds==4:
        preds="Tomato Septoria Leaf Spot"
    elif preds==5:
        preds="Tomato Spider mites Two-spotted Spider Mite"
    elif preds==6:
        preds="Tomato Target Spot"
    elif preds==7:
        preds="Tomato Yellow Leaf Curl Virus"
    elif preds==8:
        preds="Tomato Mosaic Virus"
    elif preds==9:
        preds="Healthy"

    return preds 


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file'] 

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # removing the background
        inpt=Image.open(file_path)
        if(".jpg" in f.filename):
            temp=f.filename.replace(".jpg",".png")
        else:
            temp=f.filename
        new_path=os.path.join(basepath,'uploads',secure_filename(temp))
        inpt.save(new_path)
        inpt=Image.open(new_path)
        output=remove(inpt)
        """ fpp=str(file_path)
        if(".jpg" in fpp):
            print(fpp)
            fpp.replace(".jpg",".png") """
        output.save(new_path)
        # Make prediction
        preds = model_predict(new_path, model)
        result=preds
        return result
    return None


@app.route('/')
def home():
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)
