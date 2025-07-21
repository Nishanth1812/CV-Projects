import cv2
import mediapipe as mp
import time 
import requests
import imutils
import numpy as np



class facetrack:
    
    def __init__(self,minDetectionCon=0.5):
        
        self.minDetectionCon=minDetectionCon
        
        # Setting up the mediapipe parts
        self.mp_facedetection=mp.solutions.face_detection
        self.face_detection=self.mp_facedetection.FaceDetection(self.minDetectionCon)
        self.mp_draw=mp.solutions.drawing_utils
        
        
    def find_faces(self,img,draw=True):
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.res=self.face_detection.process(img_rgb)
    
        b_box_list=[]
        if self.res.detections:
            for id,detection in enumerate(self.res.detections):
                h,w,c=img.shape
                b_box_class=detection.location_data.relative_bounding_box
                b_box= int(b_box_class.xmin*w),int(b_box_class.ymin*h),int(b_box_class.width*w),int(b_box_class.height*h) 
                
                b_box_list.append([id,b_box,detection.score])
                if draw:
                    img=self.fancy_draw(img,b_box)
                
                    cv2.rectangle(img,b_box,(0,255,0),2)
                    cv2.putText(img,f"{int(detection.score[0]*100)}%",(b_box[0],b_box[1]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        return img,b_box_list
    
    def fancy_draw(self,img,b_box,l=20,t=7,rt=1):
        x,y,w,h=b_box
        
        x1,y1=x+w,y+h
        
        cv2.rectangle(img,b_box,(0,255,0),rt)
        
        # top left x,y
        cv2.line(img,(x,y),(x+l,y),(0,255,0),t)
        cv2.line(img,(x,y),(x,y+l),(0,255,0),t)
        
        # top right x1,y
        cv2.line(img,(x1,y),(x1-l,y),(0,255,0),t)
        cv2.line(img,(x1,y),(x1,y+l),(0,255,0),t)
        
        # bottom left x,y1
        cv2.line(img,(x,y1),(x+l,y1),(0,255,0),t)
        cv2.line(img,(x,y1),(x,y1-l),(0,255,0),t)
        
        # bottom right x1,y1
        cv2.line(img,(x1,y1),(x1-l,y1),(0,255,0),t)
        cv2.line(img,(x1,y1),(x1,y1-l),(0,255,0),t)
        
        return img
def main(url=None):
    # stream=cv2.VideoCapture(0)
    pt=0
    ct=0
    detector=facetrack()
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=640, height=480)
        # success,img=stream.read()
        img,boxes=detector.find_faces(img)
        ct=time.time()    
        fps=1/(ct-pt)
        pt=ct
        cv2.putText(img,f"FPS: {int(fps)}",(20,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("Stream",img)
        cv2.waitKey(1)
        

if __name__ =="__main__":
    main("http://192.168.80.127:8080/shot.jpg")