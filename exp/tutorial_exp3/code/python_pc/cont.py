import cv2  
import numpy as np
kernel = np.ones((1, 5), np.uint8)
img = cv2.imread("./dataset/duck/0.jpg")  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
#binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=5)
contours, hierarchy,_ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
print(contours)
cv2.drawContours(img,contours,-1,(0,0,255),3)  
cv2.imshow("img", img)  
cv2.waitKey(0)  