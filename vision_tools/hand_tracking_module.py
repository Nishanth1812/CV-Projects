import cv2 
import mediapipe as mp
import time

class handtrack:
    
    def __init__(self,mode=False,maxhands=2,detectioncon=0.5,trackcon=0.5):
        self.mode=mode
        self.maxhands=maxhands
        self.detectioncon=detectioncon
        self.trackcon=trackcon
        
        self.mphands=mp.solutions.hands
        self.hands=self.mphands.Hands(self.mode,self.maxhands,min_detection_confidence=self.detectioncon,min_tracking_confidence=self.trackcon)
        self.draw=mp.solutions.drawing_utils

    def find_landmarks(self,img,draw=True):
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.res=self.hands.process(img_rgb)
    
        if (self.res.multi_hand_landmarks):
            for lms in self.res.multi_hand_landmarks:
                if draw:
                    self.draw.draw_landmarks(img,lms,self.mphands.HAND_CONNECTIONS)
        return img
    
    def get_locations(self,img,hand_no=0,draw=True):
        locations=[]
        if self.res.multi_hand_landmarks:
            hand=self.res.multi_hand_landmarks[hand_no]
            for id,lm in enumerate(hand.landmark):
                
                h,w,c=img.shape
                x,y=int(lm.x*w),int(lm.y*h)
                locations.append([id,x,y])
                
                if draw:
                    cv2.circle(img,(x,y),10,(0,255,0),3)
        return locations
    
def main():
    stream=cv2.VideoCapture(0)
    pt=0
    ct=0
    tracker=handtrack()
    while True:
        success,img=stream.read()
        img=tracker.find_landmarks(img)
        locations=tracker.get_locations(img,False)
        
        if len(locations) != 0:
            print(locations[4])
        ct=time.time()    
        fps=1/(ct-pt)
        pt=ct
    
        cv2.putText(img,str(int(fps)),(20,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)

    
        cv2.imshow("stream",img)
    
        cv2.waitKey(1)
        

if __name__=="__main__":
    main()
            