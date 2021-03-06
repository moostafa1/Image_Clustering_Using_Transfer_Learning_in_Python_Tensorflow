import os
import cv2
import numpy as np

def stackImages(scale, imgArray):

    # get image dimensions
    rows = len(imgArray)
    cols = len(imgArray[0])

    rowsAvailable = isinstance(imgArray[0], list) # returns true if the specified object of the specified type
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2] :
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0), None, scale, scale)

                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)

                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)


        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank] * rows

        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)

    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2] :
                imgArray[x] = cv2.resize(imgArray[x], (0,0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)

            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver










if __name__ == "__main__":
    img = cv2.imread(r'D:\Courses\Me\openCV_code\data\lena.jpg')
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(type(img))

    imgStack = stackImages(0.2,([img,imgGray,img,img,img,img],[img,img,img,img,img,img],[img,img,img,img,img,img]))

    #i = cv2.imread(r'clusters\class_3\331.jpeg')

    cv2.imshow('output', imgStack)
    cv2.waitKey(0)
