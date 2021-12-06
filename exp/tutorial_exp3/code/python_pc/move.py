import cv2

label = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]

# (23,34)
count1 = 0
count2 = 0
for i in range(0,23):
    img = cv2.imread("./alldata/"+str(i)+".jpg",cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_bw = im_bw[200:400,150:550]
    if label[i] is 0:
        cv2.imwrite("./dataset/duck/"+str(count1)+".jpg", im_bw)
        count1 += 1
    else:
        cv2.imwrite("./dataset/others/"+str(count2)+".jpg", im_bw)
        count2 += 1
