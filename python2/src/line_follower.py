from line_detector import Line_Detector
from PID import PID
import cv2
import numpy as np

detector = Line_Detector()
pid = PID(2, 0, 0)

img = cv2.imread('../imgs/image0.png')

cte = detector.find_cte(img)

detect_img = detector.visualize(img)
cv2.imshow('detect', detect_img)
cv2.waitKey(0)

speed = 0
thr = 0.8
if(abs(cte) > 0.5):
    thr = 0.5
if(abs(pid.p_error - cte) > 0.1 and abs(pid.p_error - cte) <= 0.2):
    thr = 0.0
elif(abs(pid.p_error - cte) > 0.2 and speed > 30):
    thr = -0.2

#LEFT IS POSITIVE RIGHT IS NEGATIVE
pid.update_error(cte, 0.1)
steer_value = -pid.total_error()
if(steer_value > 1):
    steer_value = 1
elif(steer_value < -1):
    steer_value = -1
print(steer_value)
