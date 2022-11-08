# 1st need to download the below link in the terminal
#pip3 install --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.3.1 torchvision==0.4.2

from facial_emotion_recognition import EmotionRecognition
import cv2 as cv


er = EmotionRecognition(device='cpu', gpu_id=0)
cam = cv.VideoCapture(0)
success, frame = cam.read()
frame = er.recognise_emotion(frame, return_type='BGR')
cv.imshow("frame", frame)
while True:
    key = cv.waitKey(10)
    if key & 0xff == 27:
        break
