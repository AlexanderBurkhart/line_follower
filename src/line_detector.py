import cv2
import numpy as np
import math

class Line_Detector(object):
    def warp(self, img, type):
        src = np.float32([[400, 320],
                          [870, 320],
                          [320, 650],
                          [960, 650]])

        dst = np.float32([[0, 0],
                          [450, 0],
                          [0, 450],
                          [450, 450]])

        #perspective transform
        if type == 0:
            M = cv2.getPerspectiveTransform(src, dst)
        #reverse perspective transform
        else:
            Minv = cv2.getPerspectiveTransform(dst, src)
            return Minv

        warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
        
        return warped

    def colorSpace(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_white = np.array([0,0,100], dtype=np.uint8)
        upper_white = np.array([0,0,255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_white, upper_white)
        res = cv2.bitwise_and(img, img, mask=mask)
        return res

    def calc_distance(self, p1, p2):
        return math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)  

    def find_corners(self, rect, bound):
        #tl=0 tr=1 bl=2 br=3
        #compare the distances to each point of the max corners on the image
        points = cv2.boxPoints(rect)
        points_distances = np.zeros((4,4))
        sorted_points = np.zeros((4,2))

        #calculate the distances
        for i in range(0, len(bound)):
            for j in range(0, len(points)):
                points_distances[i][j] = self.calc_distance(points[j], bound[i])
       
        #compare each group of distances that correspond to each point: tl, tr, etc. and assign it to the certain point
        for i in range(0, len(points_distances)):
            distances = points_distances[i]
            least_dist = 10000
            least_idx = -1
            for j in range(0, len(distances)):
                if(distances[j] < least_dist):
                    least_dist = distances[j]
                    least_idx = j
            sorted_points[i] = points[least_idx]
            for j in range(0, len(points_distances)):
                    points_distances[j][least_idx] = 10000

        return sorted_points
   
    def filter_contours(self, contours):
        i = 0
        while(i < len(contours)):
            contour = contours[i]
            #print("area of contour %f: %f" % (i, cv2.contourArea(contours[i])))
            if(cv2.contourArea(contour) < 500):
                contours.pop(i)
                if(i != 0):
                    i -= 1
                continue
            i += 1
        return contours

    def find_tape_direction(self, points):
        x1 = 0
        y1 = 0
        x2 = 0 
        y2 = 0
        if(self.calc_distance(points[0], points[1]) > self.calc_distance(points[0], points[2])):
            x1 = int((points[0][0]+points[2][0])/2)
            y1 = int((points[0][1]+points[2][1])/2)
            x2 = int((points[1][0]+points[3][0])/2)
            y2 = int((points[1][1]+points[3][1])/2)
        else:
            x1 = int((points[0][0]+points[1][0])/2)
            y1 = int((points[0][1]+points[1][1])/2)
            x2 = int((points[2][0]+points[3][0])/2)
            y2 = int((points[2][1]+points[3][1])/2)
            
        return x1, y1, x2, y2

    def contours(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = self.filter_contours(contours)
        #contour_img = cv2.drawContours(img, filtered_contours, -1, (0,255,0), 3)
        return filtered_contours

    def find_lines(self, contours):
        lines = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            x,y,w,h = cv2.boundingRect(contour)
            bound = [[x,y],[x+w,y],[x,y+h],[x+w,y+h]]
            #tl=0 tr =1 bl=2 br=3
            points = self.find_corners(rect, bound)
            #cv2.line(contour_img,(int(points[0][0]),int(points[0][1])),(int(points[1][0]),int(points[1][1])),(0,0,255),5)
            x1,y1,x2,y2 = self.find_tape_direction(points)
            lines.append([x1,y1,x2,y2])
            #cv2.line(contour_img,(x1,y1),(x2,y2),(255,0,0),5)
        return lines

    def draw_lines(self, lines, warped, img):
        blank_warped = np.zeros((img.shape[1], img.shape[0], 3), np.uint8)
        for line in lines:
            x1,y1,x2,y2 = line
            cv2.line(blank_warped,(x1,y1),(x2,y2),(255,0,0),5)
        Minv = self.warp(blank_warped, 1)
        newwarp = cv2.warpPerspective(blank_warped, Minv, (img.shape[1], img.shape[0]))

        result = cv2.addWeighted(newwarp, 1, img, 0.4, 0)
        return result
        

img = cv2.imread('../imgs/image0.png')

#width=640 height=480
img = cv2.resize(img, (1280,720))
#print(img.shape[1])
detector = Line_Detector()

cv2.imshow('img', img)

warped = detector.warp(img, 0)
#cv2.imshow('warped', warped)
#cv2.waitKey(0)

imgThresh = detector.colorSpace(warped)
#cv2.imshow('thresh', imgThresh)
#cv2.waitKey(0)

contours = detector.contours(imgThresh)
lines = detector.find_lines(contours)

detect_img = detector.draw_lines(lines, warped, img)

cv2.imshow('detect_line', detect_img)
cv2.waitKey(0)
#cv2.imshow('lines', lines)
#cv2.waitKey(0)
