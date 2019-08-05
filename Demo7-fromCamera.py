# import the necessary packages
import json
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io

from utils import load_model


import sys
import argparse

pathIn = "videoin/traffic.mp4"
pathOut = "videoout/cam2/"


img_width, img_height = 224, 224
print("LOADING MODEL")
model = load_model()
print("LOADING WEIGHTS")
model.load_weights('models/model.96-0.89.hdf5')
print("FINDING CLASS")
cars_meta = scipy.io.loadmat('devkit/cars_meta')
class_names = cars_meta['class_names']  # shape=(1, 196)
class_names = np.transpose(class_names)

results = []
count = 0
vidcap = cv.VideoCapture(1)
frame_counter=0
while(True):
    #vidcap.set(cv.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
    success,bgr_img = vidcap.read()
    print ('Read a new frame: ', success)
    cv.imshow('frame',bgr_img)
    bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    rgb_img = np.expand_dims(rgb_img, 0)
    print("CHANGED IMAGE")
    preds = model.predict(rgb_img)
    print("MADE PREDICITON")
    prob = np.max(preds)
    class_id = np.argmax(preds)
    text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob))
    if prob>.5:
        results.append({'label': class_names[class_id][0][0], 'prob': '{:.4}'.format(prob)})
        cv.imwrite( pathOut + "frame%d.jpg" % count, rgb_img)     # save frame as JPEG file
        print ("wrote file - count", count)
        count = count + 1
    frame_counter = frame_counter + 1
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
vidcap.release()
cv.destroyAllWindows()

print(results)
print("total frames read:", frame_counter)
print("total cars found:", count)
with open('results.json', 'w') as file:
    json.dump(results, file, indent=4)

K.clear_session()