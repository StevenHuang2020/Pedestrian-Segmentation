#Steven
import os
import cv2
import numpy as np 
from numpy import save,load

def loadImg(file,mode=cv2.IMREAD_COLOR):
    #mode = cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE cv2.IMREAD_UNCHANGED
    return cv2.imread(file,mode)

def fileList(file):
    with open(file,'r') as srcF:        
        return srcF.readlines()

def getData(file, H, W):
    lines = fileList(file)
    size = len(lines)
    print('len=',size,type(lines))
        
    trainX = []
    trainY = []
    for _ in range(size):      
        i = lines[_].strip().split(',')
        #print('i0=',i[0])
        #print('i1=',i[1])#print(type(i),i)
        fImg = i[0]
        labelMaskImg = i[1]
        
        img = loadImg(fImg, mode=cv2.IMREAD_GRAYSCALE)
        maskImg = loadImg(labelMaskImg, mode=cv2.IMREAD_GRAYSCALE)
        assert(img is not None)
        assert(maskImg is not None)
        assert(img.shape == maskImg.shape)
        #print('img.shape=',img.shape)
        #print('maskImg.shape=',maskImg.shape)
        trainX.append(img)
        trainY.append(maskImg)
    
    trainX=np.asarray(trainX)
    trainY=np.asarray(trainY)
    return trainX,trainY

def saveDataset():
    file = r'.\res\PennFudanPed\newImages\trainImages\trainList.list'
    x,y = getData(file,H=256,W=256)
    print(x.shape)
    print(y.shape)
    
    save('./trainDB/dataSetImg.npy', x)
    save('./trainDB/dataLabelMaskImg.npy', y)
    return

def loadDataset():
    x = load(r'./trainDB/dataSetImg.npy')
    y = load(r'./trainDB/dataLabelMaskImg.npy')
    
    print(x.shape)
    print(y.shape)
    return x,y

def main():
    saveDataset()
    #loadDataset()
    pass

if __name__=='__main__':
    main()
    