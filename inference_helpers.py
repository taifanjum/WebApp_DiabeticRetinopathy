import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras import backend as K


def resize_and_crop(path,scale_val):
    '''
    Reads a path file and returns the cropped images
    
    INPUT
        path: filepath of image
        resize_dim: resize image to this size (add later)
    
    OUTPUT
        img: the image loaded as rgb and resized 
    '''
    
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    y,x,channel = img.shape
    startx = x//2-(scale_val//2)
    starty = y//2-(scale_val//2)
    img = img[starty:starty+scale_val,startx:startx+scale_val]
    img_rz=cv2.resize(img,(512, 512),interpolation=cv2.INTER_AREA)
    return img_rz



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

def inference(img, model):
    '''
    Accepts a single input image of rank-4 and makes predictions and 
    returns the predicted label
    
    INPUT
        img: the image to be infered by the model
        model: the model used for inference
    
    OUTPUT
        something...
    '''
    
    img = tensor_3to4(img)
    y_pred = model.predict(img)
    y = np.argmax(y_pred)
    return y_pred, y




