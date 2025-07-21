import mediapipe as mp  # type: ignore
import cv2
import math

class HandDetector:
    def __init__(self,mode=False,max_hands=2,detection_Con=0.5,minTrack_Con=0.5):
        self.mode=mode
        self.max_hands=max_hands
        self.detection_Con=detection_Con
        self.minTrack_Con=minTrack_Con
        
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_Con,
            min_tracking_confidence=self.minTrack_Con
        ) 
        
        
        self.mpdraw=mp.solutions.drawing_utils
        self.tip_ids=[4,8,12,16,20]
        self.fingers=[]
        self.lmlist=[]
        
    def find_hands(self,img,draw=True,flip_type=True):
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(img_rgb)
        all_hands=[]
        h,w,c=img_rgb.shape
        
        if self.results.multi_hand_landmarks:
            for handType,handlms in zip(self.results.multi_handedness,self.results.multi_hand_landmarks):
                my_hand={}
                my_lmlist=[]
                xList=[]
                yList=[] 
                
                for id,lm in enumerate(handlms.landmark):
                    px,py,pz=int(lm.x*h),int(lm.y*w),int(lm.z*w)
                    my_lmlist.append([px,py,pz])
                    xList.append(px)
                    yList.append(py)
                    
                
                x_min,x_max=min(xList),max(xList)
                y_min,y_max=min(yList),max(yList)
                box_w,box_h=x_max-x_min,y_max-y_min 
                bbox=x_min,y_min,box_w,box_h
                cx,cy=bbox[0]+(bbox[2]//2),bbox[1]+(bbox[3]//2)
                
                my_hand["lmlist"]=my_lmlist
                my_hand["bbox"]=bbox
                my_hand["center"]=(cx,cy)
                
                
                if flip_type:
                    if handType.classification[0].label=='Right':
                        my_hand["type"]="Left"
                    else:
                        my_hand["type"]="Right"
                else:
                    my_hand["type"]=handType.classification[0].label
                all_hands.append(my_hand)
                
                
                if draw:
                    self.mpdraw.draw_landmarks(
                        img,handlms,self.mpHands.HAND_CONNECTIONS
                    )
                
        if draw:
            return all_hands,img
        else:
            return all_hands,img
            
    
    def fingers_up(self,my_hand):
        my_handType=my_hand["type"]
        my_lmlist=my_hand["lmlist"]
        
        if self.results.multi_hand_landmarks:
            fingers=[]
            
            # logic for thumb
            if my_handType=="Right":
                if my_lmlist[self.tip_ids[0]][0]> my_lmlist[self.tip_ids[0]-1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                
            else:
                if my_lmlist[self.tip_ids[0]][0]< my_lmlist[self.tip_ids[0]-1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0) 
            
            # logic for other hands 
            for id in range(1,5):
                if my_lmlist[self.tip_ids[id]][0]< my_lmlist[self.tip_ids[id]-2][0]:
                    fingers.append(1)
                else:
                    fingers.append(0) 
        return fingers
    
    
    def find_dist(self,p1,p2,img=None):
        x1,y1=p1
        x2,y2=p2
        cx,cy=(x1+x2)//2,(y1+y2)//2
        
        length=math.hypot(x2-x1,y2-y1)
        
        info=(x1,y1,x2,y2,cx,cy)
        
        if img is not None:
            cv2.circle(img,(x1,y1),15,(150,0,150),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(150,0,150),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
            cv2.circle(img,(cx,cy),15,(150,0,150),cv2.FILLED)
            return length,info,img
        else:
            return length,info 
        
    
    
    def find_joints(self,img,hand_no=0,draw=True):
        joints=[] 

        if self.results.multi_hand_landmarks:
            my_hand= self.results.multi_hand_landmarks[hand_no] 
            
            for id, joint in enumerate(my_hand.landmark):
                h,w,c=img.shape 
                cx,cy=int(joint.x*w), int(joint.y*h)
                joints.append([id,cx,cy]) 
                
                if draw:
                    cv2.circle(img,(cx,cy),8,(150,0,150),cv2.FILLED)
        
        return joints 
    
