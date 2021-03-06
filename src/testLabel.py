#python3 steve
#10/04/2020 Penn-Fudan dataset augmentation
#resizing and cropping
import sys
import cv2
import numpy as np 
from modules.folder.folder_file import getFileName,createPath
from genImageBoxLabel import listFile
from commonModule.ImageBase import *


def getLabelFileLabels(fileLabel):
    labels = []
    with open(fileLabel,'r') as srcF:
        for i in srcF.readlines():
            clas,x,y,w,h = i.split()   
            labels.append((x,y,w,h))
    return labels

def getCoordinatesFromLabels(H,W,fileLabel):
    coords = []
    with open(fileLabel,'r') as srcF:
        for i in srcF.readlines():
            clas,x,y,w,h = i.split()
            x,y,w,h = float(x),float(y),float(w),float(h)
            #print(x,y,w,h,H,W)
            #Xmin,Ymin,Xmax,Ymax = int(x*W), int(y*H), int(x*W) + int(w*W), int(y*H + h*H)
            xCenter,yCenter,weight,height = x*W, y*H, w*W, h*H
            Xmin = int(xCenter - weight*0.5)
            Ymin = int(yCenter - height*0.5)
            Xmax = int(xCenter + weight*0.5)
            Ymax = int(yCenter + height*0.5)
            #print('i=',i,'cod:',Xmin,Ymin,Xmax,Ymax)
            coords.append((Xmin,Ymin,Xmax,Ymax))
    return coords

def rectangleImgFromCoordinates(img,coords):
    for i in coords:
        Xmin,Ymin,Xmax,Ymax = i
        img = rectangleImg(img,(Xmin,Ymin),(Xmax,Ymax))
    return img

def testFileLabel(imgPath,LabelPath,dstRecImgPath):
    createPath(dstRecImgPath)
    for i in listFile(imgPath):
        imgFile = getFileName(i)
        img = loadImg(i)
        H,W = getImgHW(img)
        print(i,imgFile,H,W)
        
        labelFile = LabelPath + '\\' + imgFile[:imgFile.rfind('.')] +'.txt'
        #labels = getLabelFileLabels(labelFile)
        #print(labels)
        coordinats = getCoordinatesFromLabels(H,W,labelFile)
        #print(coordinats)
        recImg = rectangleImgFromCoordinates(img.copy(),coordinats)
        
        destFile = dstRecImgPath + '\\' + imgFile[:imgFile.rfind('.')]+'_rec' + '.png'
        print(destFile)
        writeImg(recImg, destFile)
        #break
    
def main():
    #base = r'.\res\PennFudanPed\trainEx\\'
    #base = r'.\res\PennFudanPed\\'
    base = r'.\res\PennFudanPed\trainExSmall\\'
    imgPath = base + 'test_PNGImages'
    LabelPath = base + r'labels\\test_PNGImages'
    dstRecImgPath = base + r'test'
        
    testFileLabel(imgPath,LabelPath,dstRecImgPath)
           
if __name__ == '__main__':
    main()