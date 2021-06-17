# USAGE
# 

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from imutils import paths


def mask_image():
    # construct the argument parser and parse the arguments
    img_dir = "/home/lamia/Downloads/test_input_saudi"
    img_paths = []
    img_paths += [el for el in paths.list_images(img_dir)]
    num = len(img_paths)
    model = load_model("/home/lamia/Downloads/plate_3type.h5")
    count=0
    print("%d pics in total" % num)
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    for annotation in img_paths:      
        im_path = annotation
        im_path = annotation
        basename = os.path.basename(im_path)
        imgname, suffix = os.path.splitext(basename)
        print('imgname',imgname)
        plate1= cv2.imread(im_path)
        plate=plate1
        #plate = cv2.imread("")
    			# extract the plate ROI, convert it from BGR to RGB channel
    			# ordering, resize it to 224x224, and preprocess it
        #plate = image[startY:endY, startX:endX]
        #cv2.imshow("cropped", plate)
        #cv2.waitKey(0)
        print('hiii')
        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
        plate = cv2.resize(plate, (224, 224))
        plate = img_to_array(plate)
        plate = preprocess_input(plate)
        plate = np.expand_dims(plate, axis=0)
    			# pass the plate through the model to determine if the plate
    			# has a mask or not
        (gcc, saudi) = model.predict(plate)[0]
    
    			# determine the class label and color we'll use to draw
    			# the bounding box and text
        label = "gcc" if gcc > saudi else "saudi"
        color = (0, 255, 0) if label == "gcc" else (0, 0, 255)
    
    			# include the probability in the label
        label = "{}: {:.2f}%".format(label, max(gcc, saudi) * 100)
        cv2.imwrite('/home/lamia/Downloads/test_output_gpus/'+imgname+'_'+label+'.jpg' , plate1)
    			# display the label and bounding box rectangle on the output
    			# frame
        count=count+1    
        #cv2.putText(plate, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        #cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    
    	# show the output image
        #cv2.imshow("Output", image)
        #cv2.waitKey(0)
	
if __name__ == "__main__":
	mask_image()
