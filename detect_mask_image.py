# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from imutils import paths
#mask_detector 12  mask_detector_32 13
#mask_detector15sep 19
def mask_image():
    # construct the argument parser and parse the arguments
    img_dir = "C:/Users/Sara/Desktop/mtcnn-tf-plate-detection-master/dataset/output11"
    img_paths = []
    img_paths += [el for el in paths.list_images(img_dir)]
    num = len(img_paths)
    model = load_model("C:/Users/Sara/Desktop/Face-Mask-Detection-master/mask_detector_32.model")
    count=0
    print("%d pics in total" % num)  
    for annotation in img_paths:      
        im_path = annotation
        im_path = annotation
        basename = os.path.basename(im_path)
        imgname, suffix = os.path.splitext(basename)
        print('imgname',imgname)
        face1= cv2.imread(im_path)
        face=face1
        #face = cv2.imread("C:/Users/Sara/Desktop/mtcnn-tf-plate-detection-master/dataset/output/hi.jpg")
    			# extract the face ROI, convert it from BGR to RGB channel
    			# ordering, resize it to 224x224, and preprocess it
        #face = image[startY:endY, startX:endX]
        #cv2.imshow("cropped", face)
        #cv2.waitKey(0)
        print('hiii')
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
    			# pass the face through the model to determine if the face
    			# has a mask or not
        (mask, withoutMask) = model.predict(face)[0]
    
    			# determine the class label and color we'll use to draw
    			# the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
    			# include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.imwrite('C:/Users/Sara/Desktop/mtcnn-tf-plate-detection-master/dataset/output_mask/'+imgname+'_'+label+'.jpg' , face1)
    			# display the label and bounding box rectangle on the output
    			# frame
        count=count+1    
        #cv2.putText(face, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        #cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    
    	# show the output image
        #cv2.imshow("Output", image)
        #cv2.waitKey(0)
	
if __name__ == "__main__":
	mask_image()
