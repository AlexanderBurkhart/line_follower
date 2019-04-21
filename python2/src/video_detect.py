import cv2
from line_detector import Line_Detector

video = cv2.VideoCapture("../vid/test_video.mp4")

detector = Line_Detector()

while True:
    (grabbed, frame) = video.read()

    if not grabbed:
        break
    detect_frame = detector.visualize(frame) 
    cv2.imshow('frame', detect_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
