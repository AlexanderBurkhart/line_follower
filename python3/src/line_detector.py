import cv2
import numpy as np
import math
from numpy import linalg as LA

class Line_Detector(object):
    def warp(self, img, type):
        src = np.float32([[450, 320],
                          [920, 320],
                          [370, 650],
                          [910, 650]])

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

    def increase_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def colorSpace(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
        #hsv
        sens = 80
        lower_white = np.array([0,0,255-sens], dtype=np.uint8)
        upper_white = np.array([255,sens,255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_white, upper_white)
        res = cv2.bitwise_and(img, img, mask=mask)

        lower_white = np.array([230,230,230], dtype=np.uint8)
        upper_white = np.array([255,255,255], dtype=np.uint8)

        mask = cv2.inRange(res, lower_white, upper_white)
        res = cv2.bitwise_and(res, res, mask=mask)

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
        if not contours:
            return []
        min_contour_area = 700
        i = 0
        while(i < len(contours)):
            contour = contours[i]
            x1,y1,x2,y2 = self.find_line(contour)
            length = self.calc_distance((x1,y1),(x2,y2))
            if cv2.contourArea(contour) < min_contour_area or length < 100:
                contours.pop(i)
                i -= 1
            i += 1

        return contours

    def find_tape_direction(self, points):
        x1 = 0
        y1 = 0
        x2 = 0 
        y2 = 0
        if self.calc_distance(points[0], points[1]) > self.calc_distance(points[0], points[2]):
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

    def find_line(self, contour):
        rect = cv2.minAreaRect(contour)
        x,y,w,h = cv2.boundingRect(contour)
        bound = [[x,y],[x+w,y],[x,y+h],[x+w,y+h]]
        #tl=0 tr =1 bl=2 br=3
        points = self.find_corners(rect, bound)
        x1,y1,x2,y2 = self.find_tape_direction(points)
        return x1,y1,x2,y2

    def closest_line(self, contours, img):
        min_cte = 10000
        min_idx = -1
        for i in range(0, len(contours)):
            contour = contours[i]
            line = self.find_line(contour)
            cte = self.calc_cte(line, img)
            if cte < min_cte:
                min_cte = cte
                min_idx = i
        return self.find_line(contours[min_idx])

    def calc_cte(self, line, img):
	#dot product
        x1,y1,x2,y2 = line
        p1 = [x1,450-y1]
        p2 = [x2,450-y2]
        p3 = [225, 450]

        dy = p2[1]-p1[1]	
        dx = p2[0]-p1[0]
        if(dx == 0): return p2[0]-p3[0]

        a = -(dy)/(dx)
        b = 1
        c = -a*p1[0]-b*p1[1]
	
        is_right = False

        if p1[0] > 225 or p2[0] > 225: is_left = True

        #print(img.shape[0])
        d = abs(a*p3[0] + b*p3[1] + c)/math.sqrt(a**2+b**2)        

        if is_right: d *= -1

        return d

    def detect_line(self, img, type):
        #img = self.increase_brightness(img, value=20)
        warped = self.warp(img, 0)
        imgThresh = self.colorSpace(warped)

#        cv2.imshow('thresh', imgThresh)

        contours = self.contours(imgThresh)
        if len(contours) == 0:
            if type == 1:
                return None, None
            return None

        line = self.closest_line(contours, img)

        if type==1:
            return line, warped

        return line

    def find_cte(self, img):
        img = cv2.resize(img, (1280,720))
        line = self.detect_line(img, 0)

        if line == None:
            return None

        return self.calc_cte(line, warped)

    def visualize(self, img): 
        img = cv2.resize(img, (1280,720))
        line, warped = self.detect_line(img, 1)

        if line == None:
            return img

        detect_img = self.draw_line(line, warped, img)

        print('cte: %f' % self.calc_cte(line, warped))

        return detect_img


    def draw_line(self, line, warped, img):
        blank_warped = np.zeros((img.shape[1], img.shape[0], 3), np.uint8)
        x1,y1,x2,y2 = line
        cv2.line(blank_warped,(x1,y1),(x2,y2),(255,0,0),5)
        Minv = self.warp(blank_warped, 1)
        newwarp = cv2.warpPerspective(blank_warped, Minv, (img.shape[1], img.shape[0]))

        result = cv2.addWeighted(newwarp, 1, img, 0.4, 0)
        return result
