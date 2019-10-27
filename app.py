from __future__ import division, print_function
# coding=utf-8
import sys
import cv2
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras import backend as K




#<!-- helper function --!>

def check_units(y_true, y_pred):
    '''
    I don't know what it does exactly
    '''

    if y_pred.shape[1] != 1:
        y_pred = y_pred[:,1:2]
        y_true = y_true[:,1:2]
    return y_true, y_pred


def f1(y_true, y_pred):
    '''
    Calculates the F1 score
    '''

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_true, y_pred = check_units(y_true, y_pred)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def get_data(path):
    '''
    Reads a path file and returns the images

    INPUT
        path: filepath of image
        resize_dim: resize image to this size (add later)

    OUTPUT
        img: the image loaded as rgb and resized
    '''

    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(512, 512),interpolation=cv2.INTER_AREA)
    return img


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
from keras.applications.resnet50 import ResNet50
# weights = "imagenet" indicates that we want to use the pre-trained ImageNet weights for the respective model.
#model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_data(path):
    '''
    Reads a path file and returns the images

    INPUT
        path: filepath of image
        resize_dim: resize image to this size (add later)

    OUTPUT
        img: the image loaded as rgb and resized
    '''

    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(512, 512),interpolation=cv2.INTER_AREA)
    return img



def model_predict(img_path, model):
    # load the image given in img_path and resize it to the required size using target_size
    #img = image.load_img(img_path, target_size=(512, 512))
    img = get_data(img_path)
    # Preprocessing the image: convert the pixels to a NumPy array
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    # Expand the shape of an array. x = array, axis = Position in the expanded axes where the new axis is placed (0 for x axis and 1 for y axis)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # preprocess x with caffe framework
    x = preprocess_input(x, mode='caffe')
    # run the prediction model on x
    preds = model.predict(x)
    # return prediction value
    return preds

def tensor_3to4(img):
    '''
    Expands dimension.

    INPUT:
        img: a 3-channel image as input

    OUTPUT
        A rank-4 tensor, since the network accepts batches of images. One image corresponds to batch size of 1
    '''
    img_4d = np.expand_dims(img, axis=0)  # rank 4 tensor for prediction
    return img_4d

def inference(x, model):
    '''
    Accepts a single input image of rank-4 and makes predictions and
    returns the predicted label

    INPUT
        img: the image to be infered by the model
        model: the model used for inference

    OUTPUT
        something...
    '''
    img = get_data(x)


    img = tensor_3to4(img)



    y_pred = model.predict(img)
    print(y_pred)

    y = np.argmax(np.array(y_pred))

    return y_pred, y

@app.route('/', methods=['GET'])
def index():
    # Main page
    # initiate index.html in browser
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        print("debug korbo!")
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)
        print(file_path)
        # predict class labels
        preds, pred_arg = inference(file_path, model)
        print(preds, pred_arg)

        # Process your result for human
        #pred_class = preds.argmax(axis=-1)            # Simple argmax

        # gives us the ImageNet Unique ID of the label, along with a human-readable text version of the label
        #pred_class = decode_predictions(preds, top=1)
        #print(type(pred_class))   # ImageNet Decode
        # Convert to string
        #result = str(pred_class[0][0][1])
        if pred_arg==0:
            result = 'Its Normal Bro! Congratulations'
        else:
            result = 'Its Diabetic Retinopathy Bro!'
        return result
    return None


if __name__ == '__main__':
    #app.run(port=5000, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
