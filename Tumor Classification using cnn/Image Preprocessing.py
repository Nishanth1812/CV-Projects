# This file is used to preprocess the images present in the dataset before performing data augmentation 

# Importing the necessary modules
import cv2
import os
import imutils
import numpy as np
import matplotlib.pyplot as plt 
import time 


# Creating a function to crop the images to the appropriate size to improve model performance 

def crop_img(img):
    
    # Converting image to proper GrayScale
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Adding a slight blur to it
    gray=cv2.GaussianBlur(gray,(5,5),0) 
    
    # Thresholding the image (i.e converting it into a binary image)
    
    thresh=cv2.threshold(gray,45,255,cv2.THRESH_BINARY)[1]
    thresh=cv2.erode(thresh,None,iterations=2)
    thresh=cv2.dilate(thresh,None,iterations=2) 
    
    
    # Finding contours of the binary image and choosing the largest one to crop it later 
    
    cont=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cont=imutils.grab_contours(cont)
    cnts=max(cont, key=cv2.contourArea) 
    
    # Finding the extreme points of the contours 
    left=tuple(cnts[cnts[:,:,0].argmin()][0])
    right=tuple(cnts[cnts[:,:,0].argmax()][0])
    top=tuple(cnts[cnts[:,:,1].argmin()][0])
    bot=tuple(cnts[cnts[:,:,1].argmax()][0]) 
    
    new_img=img[top[1]:bot[1],left[0]:right[0]]
    
    return new_img 


def load_data(dir,img_size,save_dir):
    
    img_width,img_height=img_size
    
    for i in range(len(dir)):
        
        in_dir=dir[i]
        out_dir=save_dir[i] 
        
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"The output directory of the file does not exist {save_dir}")
        
        for file in os.listdir(dir):
            input_path=os.path.join(dir,file)
            out_path=os.path.join(save_dir,file)
            
            img=cv2.imread(input_path)
            
            img=crop_img(img)
            img=cv2.resize(img,(img_width,img_height), interpolation=cv2.INTER_CUBIC)
            
            cv2.imwrite(out_path,img)
    print("Processed image saved")

# To calculate the amount of time elapsed 

def time_str(t_elap):
    hrs=int(t_elap/3600) # to get number of hours
    min_=int((t_elap%3600)/60) #minutes are calculated from the seconds left after accounting for full hours
    sec_=int(t_elap%60) #seconds are calculated from the time left after accounting for full minutes 
    
    return f"{hrs}:{min_}:{sec_}"



# Running the crop function on the dataset images

# Input paths
yes_in_path=r"Data\yes"
no_in_path=r"Data\no"

# Output paths
yes_out_path=r"Preprocessed Data\Yes"
no_out_path=r"Preprocessed Data\No"

#Required Image size

img_w=240
img_h=240

start=time.time()
load_data(yes_in_path,(img_w,img_h),yes_out_path)
load_data(no_in_path,(img_w,img_h),no_out_path)

end=time.time() 

e_time=(end-start)

print(f"Total time elaspsed: {time_str(e_time)}")