from houfline import houfline
import cv2 as cv
import numpy as np
import math
# for i in range(1,9):

#     img = cv.imread("E:\\code\\data\\0"+str(i)+".jpg")
#     print("E:\\code\\data\\0"+str(i)+".jpg")
#     # cv.line(img,(0,240),(640,240),(0,0,255),2)
#     # cv.imshow('img',img)
#     # cv.waitKey(0)
#     # cv.destroyAllWindows()
#     list = houfline(img)

#     #print(list)
#     # #img = cv.imread("E:\\code\\data\\02.jpg")
#     # [dis,m] = houfline(img)
#     # print(dis,m)
img = cv.imread("E:\\code\\data2\\0"+str(1)+".jpg")
# #cv.line(img,(0,240),(640,240),(0,0,255),2)
# cv.line(img,(240,240),(320,320),(0,255,255),2)
# cv.line(img,(0,240),(320,320),(0,255,255),2)
img = houfline(img)
# cv.imshow('img',img)
# cv.waitKey(0)
# cv.destroyAllWindows()