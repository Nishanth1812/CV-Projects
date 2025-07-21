import cv2 
import mediapipe as mp
import time
import requests
import imutils
import numpy as np

class posetrack:
    def __init__(self, mode=False, upbody=False, detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.upbody = upbody
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=self.mode,model_complexity=1,smooth_landmarks=True,enable_segmentation=False,min_detection_confidence=self.detectioncon,min_tracking_confidence=self.trackcon)
        self.draw = mp.solutions.drawing_utils

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.res = self.pose.process(img_rgb)

        if self.res.pose_landmarks:
            if draw:
                self.draw.draw_landmarks(img, self.res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def get_positions(self, img, draw=True):
        locations = []
        if self.res.pose_landmarks:
            for id, lm in enumerate(self.res.pose_landmarks.landmark):
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                locations.append([id, x, y])
                if draw:
                    cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)
        return locations


def main(url=None):
    # stream = cv2.VideoCapture("sample.mp4")
    pt = 0
    tracker = posetrack()

    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=640, height=480)
        # success, img = stream.read()
        img = tracker.find_pose(img)
        locations = tracker.get_positions(img, draw=False)

        if len(locations) != 0:
            print(locations[1])  

        ct = time.time()
        fps = 1 / (ct - pt) 
        pt = ct

        cv2.putText(img, f'FPS: {int(fps)}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        cv2.imshow("Pose Tracking", img)



if __name__ == "__main__":
    main()
