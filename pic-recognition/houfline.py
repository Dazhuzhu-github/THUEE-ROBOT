import cv2 as cv
import numpy as np
import math
def houfline(img):
    #img = img[:,50:370]
    # cv.imshow("img",img)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    l_blue = np.array([[100,43,46]])
    h_blue = np.array([[124,255,255]])
    l_green = np.array([[40,100,100]])
    h_green = np.array([[60,200,255]])
    mask_b = cv.inRange(hsv, l_blue, h_blue)
    mask_g = cv.inRange(hsv, l_green, h_green)
    res = cv.bitwise_or(mask_b, mask_g)
    cv.imshow("test", res)
    cv.imshow("mask", mask_g)
    list1 = []
    edges = cv.Canny(res,50,150,apertureSize = 3)
    cv.imshow("mask2", edges)
    lines = cv.HoughLinesP(mask_g,1,np.pi/180,100,minLineLength=100,maxLineGap=1)
    lines = cv.HoughLinesP(edges,1,np.pi/180,70,minLineLength=10,maxLineGap=100)
    if lines is None:
        return 0,0,0
    for line in lines:
        x1,y1,x2,y2 = line[0]
        print(x1,y1,x2,y2)
        m = (y1-y2)/(x1-x2)
        p = [0,240]
        a = y1-y2
        b = x2-x1
        c = x1*y2 - y1*x2
        dis = abs(a*p[0]+b*p[1]+c)/math.sqrt(a*a+b*b)
        print(dis,m)
        #聚类
        flag = 0
        if (len(list1) == 0) :
            if(math.isinf(m)):
                break
            list1.append([m,dis])
        else:
            for i in range(0,len(list1)):
                if ( abs(math.atan(m)-math.atan(list1[i][0])) < 0.5 and abs(dis-list1[i][1])<200):
                    # list1[i][0] = (m+list1[i][0])/2
                    # list1[i][1] = (list1[i][1]+dis)/2
                    flag = 1
                    break
                elif(math.isinf(m)):
                    flag = 1
                    break
                    
            if (flag == 0):               
                list1.append([m,dis])
                     
            
        cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    #cv.imwrite('houghlines3.jpg',img)
    
    cv.imshow("img",img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    #shape (640,480 3)
    print(list1)
    print(len(list1))
    return list1

def houfline_black(img):
    img = img[:,50:370]
    cv.imshow("img1",img)
    #hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    l_black = np.array([[0,0,0]])
    h_black = np.array([[50,100,50]])
    mask = cv.inRange(img, l_black, h_black)
    cv.imshow("mas",mask)
    edges = cv.Canny(mask,50,150,apertureSize = 3)
    cv.imshow("mask", edges)
    cv.waitKey(0)
    cv.destroyAllWindows()
    #lines = cv.HoughLinesP(mask_g,1,np.pi/180,100,minLineLength=100,maxLineGap=1)
    lines = cv.HoughLinesP(edges,1,np.pi/180,70,minLineLength=10,maxLineGap=100)
    x1,y1,x2,y2 = lines[0][0]
    # if lines is None:
    #     return 0,0,0
    # for line in lines:
    #     x1,y1,x2,y2 = line[0]
    #     cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    m = (y1-y2)/(x1-x2)
    p = [-50,240]
    a = y1-y2
    b = x2-x1
    c = x1*y2 - y1*x2
    dis = abs(a*p[0]+b*p[1]+c)/math.sqrt(a*a+b*b)
    return [m,dis]
    # cv.imshow("final",img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # return 0
