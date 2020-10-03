import cv2
import numpy as np
 
def clipImg(img,startPt,stopPt):
    return img[startPt[1]:stopPt[1], startPt[0]:stopPt[0]]

def blurImg(img,ksize=5): #Averaging adjency pixsel 5x5 size kernel
    return cv2.blur(img, (ksize,ksize))

def gaussianBlurImg(img, ksize=5): #Gaussian Blurring
    kernel = cv2.getGaussianKernel(ksize,0)
    #print('gaussian kernel=',kernel)
    return cv2.GaussianBlur(img,(ksize,ksize),0)

def medianBlurImg(img,ksize=5): #Median Blurring
    return cv2.medianBlur(img, ksize)

def adjustBrightnessAndContrast(img,alpha=0.1,beta=100):
    """g(x) = alpha * f(x) + beta, alpha(0.1~3) beta(0~100)"""
    if 1:
        new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    else:
        new_image = np.zeros_like(img)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                for c in range(img.shape[2]):
                    new_image[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)
    return new_image

def GammaCorrection(img,gamma=0.5): #gama = [0.04~25]
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(img, lookUpTable)