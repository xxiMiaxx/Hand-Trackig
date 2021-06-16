#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 20:32:57 2020

@author: lamia
"""

# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import paths
import os
import cv2
import time
#%matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np

img_dir = "/home/lamia/Documents/test_gcc_classfier/saudi/"
img_paths = []
img_paths += [el for el in paths.list_images(img_dir)]
num = len(img_paths)

#true_plates=[]
#plates=[]
#Creating label arrays for confusion metrix
Saudi_dir = "/home/lamia/Documents/test_gcc_classfier/saudi/"
saudi_paths = []
saudi_paths += [el for el in paths.list_images(Saudi_dir)]


#import numpy as np
#all_paths = np.concatenate((img_paths, saudi_paths))

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img

# load an image and predict the class
def run_example():
	# load the image
    count=1
    model = load_model('/home/lamia/Documents/test_gcc_classfier/mobileNet_V2/final_2_model.h5')
    #print(model.summery())
    t1 = time.time()
    for annotation in img_paths: 
        #print(annotation)
        #print('image number: ', count)
        im_path = annotation
        im_path = annotation
        basename = os.path.basename(im_path)
        imgname, suffix = os.path.splitext(basename)
        #print('imgname',imgname)
        #plate1= cv2.imread(im_path)
        #plate=plate
        img = load_image(annotation)
	# load model
        
	# predict the class
        result = model.predict(img)
        #print("result", result[0])
        if result[0] <0.9:
            label = "gcc"
            print("GCC PLATE")
            #plates.append(0)
            #print(plates)
        else:
            label = "saudi"
            print("SAUDI PLATE")
            #plates.append(1)
        count=count+1
        plate1= cv2.imread(im_path)
        print(cv2.imwrite("/home/lamia/Documents/test_gcc_classfier/mobileNet_V2/output_saudi/"+imgname+'_'+label+'.jpg' , plate1))
    # Testing has finished 
    t2 = time.time()
    print( 'Time taken was {} seconds'.format( t2 - t1))
    






# entry point, run the example
run_example()
