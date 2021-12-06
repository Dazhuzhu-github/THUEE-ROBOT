import cv2
class_name = ['pikaqiu', 'miaowazhongzi', 'jienigui', 'xiaohuolong', 'yibu', 'kabishou']

for j in range(0,6):
    classes = class_name[j]
    for i in range(0,7):
        img = cv2.imread("./"+classes+"/"+str(i)+".jpg")
        H, W,_ = img.shape
        cutted = img[round(H*0.1):round(H*0.9),round(W*0.6):,:]
        print(cutted.shape)
        cv2.imwrite("./"+classes+"/"+str(i+12)+".jpg",cutted)