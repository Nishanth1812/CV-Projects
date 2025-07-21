import cv2
import mediapipe as mp
import time 
import requests
import imutils
import numpy as np

class face_mesh:
    
    def __init__(self,staticMode=False,maxFaces=2,minDetectionCon=0.5,minTrackCon=0.5,thickness=1,circle_radius=2):
        
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.minDetectionCon=minDetectionCon
        self.minTrackCon=minTrackCon
        self.thickness=thickness
        self.circle_radius=circle_radius
        
        self.mpdraw=mp.solutions.drawing_utils
        self.mp_facemesh=mp.solutions.face_mesh
        self.face_mesh=self.mp_facemesh.FaceMesh(static_image_mode=self.staticMode,max_num_faces=self.maxFaces,min_detection_confidence=self.minDetectionCon,min_tracking_confidence=self.minTrackCon)  
            
        self.draw_specs=self.mpdraw.DrawingSpec(thickness=self.thickness,circle_radius=self.circle_radius)
        
        
    def find_facemesh(self,img,draw=True):
        
        self.img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.res=self.face_mesh.process(self.img_rgb)

        faces=[]
        if self.res.multi_face_landmarks:
            
            for face_lm in self.res.multi_face_landmarks:
            
                if draw:
                    self.mpdraw.draw_landmarks(img,face_lm,self.mp_facemesh.FACEMESH_CONTOURS,self.draw_specs,self.draw_specs)

                face_marks=[]
                for id,lm in enumerate(face_lm.landmark):
                    h,w,c=img.shape
                
                    x,y=int(lm.x*w),int(lm.y*h)
                    face_marks.append([x,y])
                    cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                faces.append(face_marks)
        return img,faces


def main(url=None):
    # stream=cv2.VideoCapture(0)
    pt=0
    ct=0
    detector=face_mesh()
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=640, height=480)
        # success,img=stream.read()
        
        img,faces=detector.find_facemesh(img)
        
        if len(faces)!=0:
            print(len(faces))
        
        
        
        ct=time.time()    
        fps=1/(ct-pt)
        pt=ct
        cv2.putText(img,f"FPS: {int(fps)}",(20,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("Stream",img)
        cv2.waitKey(1)
        

if __name__ =="__main__":
    main("http://192.168.80.127:8080/shot.jpg")