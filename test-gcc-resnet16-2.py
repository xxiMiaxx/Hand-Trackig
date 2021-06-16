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
import csv 

img_dir = "/home/lamia/Documents/sara2/without/"
img_paths = []
img_paths += [el for el in paths.list_images(img_dir)]
num = len(img_paths)

#Creating label arrays for confusion metrix
Saudi_dir = "/home/lamia/Documents/test_gcc_classfier/saudi/"
saudi_paths = []
saudi_paths += [el for el in paths.list_images(Saudi_dir)]
true_plates=[]

#for img_file in img_paths:
    #true_plates.append(0) 


#for img_file in saudi_paths:
    #true_plates.append(1)


#import numpy as np
#all_paths = np.concatenate((img_paths, saudi_paths))

plates=[]
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
    # load model
    model = load_model('/home/lamia/Documents/test_gcc_classfier/resnet_16/final_mask_model.h5')
    t1 = time.time()
    with open('gcc_classfier_results.csv','a',newline='') as file:
        writer= csv.writer(file)
        #writer.writerow(['image_name', 'conf','label'])
        for annotation in img_paths: 
            print(annotation)
            print('image number: ', count)
            im_path = annotation
            im_path = annotation
            basename = os.path.basename(im_path)
            imgname, suffix = os.path.splitext(basename)
            print('imgname',imgname)
            plate1= cv2.imread(im_path)
            #plate=plate
            img = load_image(annotation)
    	
    	# predict the class
            result = model.predict(img)
            print("result", result[0])
            if result[0] <= 0.5:
                label = "with"
                print("with mask")
                print(cv2.imwrite("/home/lamia/Documents/sara2/with_results/"+imgname+'_'+label+'_conf:'+str(result[0])+'.jpg' , plate1))
                writer.writerow([imgname, str(result[0]),label])
                #plates.append(0)
                #plates.append(annotation)
            else:
                label = "without"
                print("without mask")
                print(cv2.imwrite("/home/lamia/Documents/sara2/without_results/"+imgname+'_'+label+'_conf:'+str(result[0])+'.jpg' , plate1))
                writer.writerow([imgname, str(result[0]),label])
                #plates.append(1)
            count=count+1
            #plate1= cv2.imread(im_path)
        
    # Testing has finished 
    t2 = time.time()
    print( 'Time taken was {} seconds'.format( t2 - t1))
    
    











# entry point, run the example
run_example()