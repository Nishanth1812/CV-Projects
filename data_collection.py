import cv2
import time
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
from math import *



pt=0
ct=0
w_cam, h_cam = 640, 480
offset=30 #used to get little more space other than the hands in the cropped image 
img_size=300 #size of the bg image used for padding
folder=r"Data\plus"
counter=0

# Setting up the video feed

stream=cv2.VideoCapture(0)
stream.set(3,w_cam)
stream.set(4,h_cam)
detector=HandDetector(maxHands=1)

# Main Loop
while True:
    success,img=stream.read()

    # detecting the hands using cvzone 
    
    hands,img=detector.findHands(img)
    
    # Cropping the detected image to remove the extra unnecessary part using try catch block
    
    try:
        
        if hands:
            hand=hands[0]
        
            # Note: if the hand is a little bit out of the view capture the script execution fails because we get an error with the imshow function 
    
            x1,y1,w1,h1=hand['bbox']
            img_crop=img[y1-offset:y1+h1+offset,x1-offset:x1+w1+offset]    
        
            # create a background image to be used for padding
            bg_img=np.ones((img_size,img_size,3),np.uint8)*255
        
            # putting the img matrix of the cropped matrix in a specific portion of the cropped image
        
            a_ratio=h1/w1 
        
            if a_ratio>1:
                k=img_size/h1
                calc_width=ceil(k*w1)
                resized_img=cv2.resize(img_crop,(calc_width,img_size))
            
                # calculating the width gap which is used to center the image
                # w_gap is the amount by which the image must be pushed to make it centered
            
                w_gap=ceil((img_size-calc_width)/2)
                bg_img[0:resized_img.shape[0],w_gap:calc_width+w_gap]=resized_img
            else:
                k=img_size/w1
                calc_height=ceil(k*w1)
                resized_img=cv2.resize(img_crop,(img_size,calc_height))
            
                # calculating the width gap which is used to center the image
                # w_gap is the amount by which the image must be pushed to make it centered
                        
                h_gap=ceil((img_size-calc_height)/2)
                bg_img[h_gap:calc_height+h_gap,0:resized_img.shape[0]]=resized_img
            
            
            cv2.imshow("background image",bg_img)
            cv2.waitKey(10)
    except Exception as e:
        pass
    
    # Dispplaying the fps on the screen
    ct=time.time()    
    fps=1/(ct-pt)
    pt=ct
    cv2.putText(img,str(f"fps:{int(fps)}"),(20,80),cv2.FONT_HERSHEY_SIMPLEX,1,(150,70,0),2)
    
    # Showing the final image 
    cv2.imshow("image",img)
    key=cv2.waitKey(1)
    if key==ord("a"):
        counter+=1
        cv2.imwrite(f"{folder}/image_{time.time()}.jpg",bg_img)
        print(counter)

