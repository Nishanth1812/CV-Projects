import pyautogui as gui  # type: ignore
import cv2 
import mediapipe as mp # type: ignore
import time
from hand_detector import HandDetector 

detector=HandDetector()
w_cam,h_cam=1280,720 
pt,ct=0,0
screen_x,screen_y=gui.size()
prev_x,prev_y=0,0
smooth=5
# Setting up stream
print(screen_x,screen_y)
stream=cv2.VideoCapture(0)
stream.set(3,w_cam)
stream.set(4,h_cam) 

# Setting up the bounding box
bound_x_min,bound_y_min=50,20
bound_x_max,bound_y_max=1200,700

bound_width=bound_x_max-bound_x_min 
bound_height=bound_y_max-bound_y_min


# MAIN LOOP
while True:
    success,img=stream.read() 
    # img=cv2.flip(img,1)
    

    # finding hands
    hands,img=detector.find_hands(img,True)
    if hands:
        hand=hands[0]
        lms=hand["lmlist"]

        # getting landmarks of thumb,index and middle fingers
    
        x_thumb,y_thumb=lms[4][0],lms[4][1] 
        x_index,y_index=lms[8][0],lms[8][1] 
        x_middle,y_middle=lms[12][0],lms[12][1] 
        # Creating the bounding box for the mouse 
        
        cv2.rectangle(img,(bound_x_min,bound_y_min),(bound_x_max,bound_y_max),(150,50,100),2)

        if (bound_x_min<=x_index<=bound_x_max and bound_y_min<=y_index<=bound_y_max): 
            
            # Chedking which fingers are up 
            fingers=detector.fingers_up(hand)
            # print(fingers)
            
            # Mouse moving mode #
    
            # Checking if index finger is up 
            if fingers[1]==1 and fingers[0]==0 and fingers[2]==0:

                # Normalising the values of the landmarks 
            
                # x_norm,y_norm=(x_index*screen_x)/bound_width,(y_index*screen_y)/bound_height
                
                x_norm,y_norm=((x_index-bound_x_min)/bound_width)*screen_x,((y_index-bound_y_min)/bound_height)*screen_y
                
                # Making the mouse movement smooth 
                
                x_smooth,y_smooth=prev_x+(x_norm-prev_x)/smooth,prev_y+(y_norm-prev_y)/smooth
                
                # Moving the mouse 
                gui.moveTo(x_norm,y_norm)
                prev_x,prev_y=x_smooth,y_smooth
                
                
            # Left click #
            
            # Checking if both thumb and index are up:
            if (fingers[0]==1 and fingers[1]==1):   
                len,info=detector.find_dist((x_thumb,y_thumb),(x_index,y_index))
                print(len)

                if len<310:
                    gui.leftClick()
            
            
            
            # Right click #
            
            # checking if both index and middle finger are up
            if fingers[1]==1 and fingers[2]==1:
                len_right,info_right=detector.find_dist((x_index,y_index),(x_middle,y_middle)) 
                
                # print(len_right)
                
                if len_right<100:
                    gui.rightClick()
                    

    # Displaying the fps
    ct=time.time()
    fps=1/(ct-pt)
    pt=ct
    
    cv2.putText(img,f"FPS: {int(fps)}",(20, 60),cv2.FONT_HERSHEY_COMPLEX,2,(100,0,150),3)
    
    # Displaying final image
    cv2.imshow("stream",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()