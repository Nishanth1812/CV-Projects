import hand_tracking_module as ht 
import cv2 
import time
# import requests
# import imutils
import numpy as np
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume # type: ignore
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL # type: ignore

###############################
# Setting the width and height of the camera capture
w_cam, h_cam = 640, 480
###############################

url = "http://192.168.80.127:8080/shot.jpg"

stream = cv2.VideoCapture(0)
stream.set(3, w_cam)
stream.set(4, h_cam)

pt = 0
ct = 0
tracker = ht.handtrack(detectioncon=0.7)

# Setting up the inits for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume.iid, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]

# main loop for video capture and processing
while True:
    ###################################
    # Code to connect phone webcam to cv2
    # img_resp = requests.get(url)
    # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    # img = cv2.imdecode(img_arr, -1)
    # img = imutils.resize(img, width=w_cam, height=h_cam)
    ###################################

    success, img = stream.read()
    img = tracker.find_landmarks(img)
    land_marks = tracker.get_locations(img, 0, False)

    # printing the landmarks
    if len(land_marks) != 0:
        x1, y1 = land_marks[4][1], land_marks[4][2]
        x2, y2 = land_marks[8][1], land_marks[8][2]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 8, (0, 255, 0), -1)
        cv2.circle(img, (x2, y2), 8, (0, 255, 0), -1)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 8, (255, 0, 255), -1)

        # Finding the length of the line
        length = math.hypot(x2 - x1, y2 - y1)

        ###############################
        # Printing the length
        # print(length)

        # putting visual changes to track the length change
        if length < 50:
            cv2.circle(img, (cx, cy), 8, (0, 0, 0), -1)
        elif length > 150:
            cv2.circle(img, (cx, cy), 8, (134, 242, 5), -1)
        ###############################

        # Hand range: 50-150
        # volume range: -65-0

        vol = np.interp(length, [30, 150], [min_volume, max_volume])
        vol_bar = np.interp(length, [50, 150], [400, 150])
        vol_percent = np.interp(length, [50, 150], [0, 100])
        print(vol)

        # Changing the volume
        volume.SetMasterVolumeLevel(vol, None)

        # Displaying the volume change visually
        cv2.rectangle(img, (50, 150), (80, 400), (253, 52, 242), 3)
        cv2.rectangle(img, (50, int(vol_bar)), (80, 400), (253, 52, 242), cv2.FILLED)
        cv2.putText(img, f'{int(vol_percent)}%', (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (253, 52, 242), 2)

    ct = time.time()
    fps = 1 / (ct - pt) if (ct - pt) != 0 else 0
    pt = ct

    cv2.putText(img, f'FPS:{int(fps)}', (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("stream", img)

