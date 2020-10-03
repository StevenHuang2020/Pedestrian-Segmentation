#python3 steve
#08/06/2020 Penn-Fudan dataset augmentation
#resizing and cropping,flipping
import sys
import cv2
import numpy as np 
from modules.folder.folder_file import createPath,getFileName,deleteFolder,copyFile
from genImageBoxLabel import getFileCoordinates,writeToAnnotFile,writeCordToAnnotFile
from genImageBoxLabel import loadImg,getImgHW,getImagChannel,listFile,getImgAnnotFile
from imageColorAugPed import ImageColorAug
from commonImage import *
from commonModule.ImageBase import *
    
def clipImgCoordinate(img,clipCoordinate,coordinates):
    clipImg = img[clipCoordinate[1]:clipCoordinate[3], clipCoordinate[0]:clipCoordinate[2]]
    
    newCoordinates=[]
    for cod in coordinates:
        Xmin,Ymin,Xmax,Ymax = cod
        newCoordinates.append((Xmin-clipCoordinate[0],Ymin-clipCoordinate[1],Xmax-clipCoordinate[0],Ymax-clipCoordinate[1]))
    return clipImg,newCoordinates

def resizeRtImg(img,ratio):
    #ratio = float(ratio)
    h,w = img.shape[0],img.shape[1]
    reH,reW = round(h*ratio), round(w*ratio)
    return resizeImg(img,reH,reW)
    #return cv2.resize(img, (reW,reH), interpolation=cv2.INTER_CUBIC) #INTER_CUBIC INTER_NEAREST INTER_LINEAR INTER_AREA

def flipImg(img):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    
    newImage = img.copy()
    for j in range(W):
        if chn>1:
            for n in range(chn):
                newImage[:, j, n] = img[:, W-j-1, n]
        else:
            newImage[:, j] = img[:, W-j-1]
    return newImage
            
def getBoundaryCoordinate(coordinates):
    x_min,y_min,x_max,y_max = coordinates[0]
    for i in coordinates:
        Xmin,Ymin,Xmax,Ymax = i
        if x_min > Xmin:
            x_min = Xmin
        if y_min > Ymin:
            y_min = Ymin
        if x_max < Xmax:
            x_max = Xmax
        if y_max < Ymax:
            y_max = Ymax
    return x_min,y_min,x_max,y_max

def generateMaskByClipping(imgPath,maskImgPath,annotPath,dstImgPath,dstMaskImgPath,N=10):
    createPath(dstImgPath)
    createPath(dstMaskImgPath)
    for i in listFile(imgPath):
        imgFile = getFileName(i)
        fAnnot = getImgAnnotFile(annotPath,i)
        img = loadImg(i)
        H,W = getImgHW(img)
        #print(fAnnot,H,W)
        name = imgFile[:imgFile.rfind('.')]
        maskImgFile = maskImgPath + '\\' + name + '_mask' + '.png'
        maskImg = loadImg(maskImgFile)
        
        coordinates = getFileCoordinates(fAnnot)
        boundCoordinate = getBoundaryCoordinate(coordinates)
        print(i,H,W,coordinates,boundCoordinate)
        
        clipXmin = np.random.randint(boundCoordinate[0], size=N)
        clipYmin = np.random.randint(boundCoordinate[1], size=N)
        clipXmax = np.random.randint(W-boundCoordinate[2], size=N)
        clipYmax = np.random.randint(H-boundCoordinate[3], size=N)
        
        for Xmin,YMin,Xmax,Ymax in zip(clipXmin,clipYmin,clipXmax,clipYmax):
            clipCoordinate = (Xmin,YMin,Xmax+boundCoordinate[2],Ymax+boundCoordinate[3])
            clipImg,_ = clipImgCoordinate(img.copy(),clipCoordinate,coordinates)
            #print(coordinates,clipCoordinate,newCoordinates,'NewH=',clipImg.shape[0],'NewW=',clipImg.shape[1])
            clipImgName = imgFile[:imgFile.rfind('.')]+'_'+str(clipCoordinate[0])+'_'+str(clipCoordinate[1])+'_'+str(clipCoordinate[2])+'_'+str(clipCoordinate[3])
            #print(clipImgName)
            writeImg(clipImg, dstImgPath + '\\' + clipImgName + '.png')
            
            clipMaskImg,_ = clipImgCoordinate(maskImg.copy(),clipCoordinate,coordinates)
            writeImg(clipMaskImg, dstMaskImgPath + '\\' + clipImgName + '_mask' +'.png')
        
def generateMaskByScaling(imgPath,maskImgPath,dstImgPath,dstMaskImgPath,rStart=0.2,rStop=2,N=20):
    createPath(dstImgPath)
    createPath(dstMaskImgPath)
    for i in listFile(imgPath):
        imgFile = getFileName(i)
        img = loadImg(i)
        #H,W = getImgHW(img)
        name = imgFile[:imgFile.rfind('.')]
        maskImgFile = maskImgPath + '\\' + name + '_mask' + '.png'
        maskImg = loadImg(maskImgFile)
        #print(i,imgFile,maskImgFile)
        
        #start generate new images
        ratios = np.linspace(rStart,rStop,N)    
        for ratio in ratios:   
            #print('ratio=',ratio)  
            rImg = resizeRtImg(img,ratio)
            newImgFile = name + '_scale_' + str(ratio)
            writeImg(rImg,dstImgPath + '\\' + newImgFile + '.png')
            
            rMaskImg = resizeRtImg(maskImg,ratio)
            newMaskImgFile = newImgFile + '_mask'
            writeImg(rMaskImg,dstMaskImgPath + '\\' + newMaskImgFile + '.png')
        
def generateMaskByFlipping(imgPath,maskImgPath,dstImgPath,dstMaskImgPath):
    createPath(dstImgPath)
    createPath(dstMaskImgPath)
    for i in listFile(imgPath):
        imgFile = getFileName(i)
        img = loadImg(i)
        #H,W = getImgHW(img)
        name = imgFile[:imgFile.rfind('.')]
        maskImgFile = maskImgPath + '\\' + name + '_mask' + '.png'
        maskImg = loadImg(maskImgFile)
        #print(i,imgFile,maskImgFile)
                
        rImg = flipImg(img)
        newImgFile = name + '_flip'
        writeImg(rImg, dstImgPath + '\\' + newImgFile + '.png')
        
        rMaskImg = flipImg(maskImg)
        newMaskImgFile = newImgFile + '_mask'
        writeImg(rMaskImg, dstMaskImgPath + '\\' + newMaskImgFile + '.png')

def handleMaskLabel(maskImgPath): #multity labels all change to 1
     for i in listFile(maskImgPath):
        img = loadImg(i)
        #c = np.unique(img)
        #print(len(c),c)
        img = np.where(img != 0, 1, 0)
        #c = np.unique(img)
        #print(len(c),c)
        writeImg(img,i)

def colorAugmentation(imgPath,maskImgPath,annotPath):
    print('Color augmentation start...')
    print('imgPath=',imgPath)
    print('maskImgPath=',maskImgPath)
    
    tmpPath = r'.\res\PennFudanPed\tmp'
    
    for i in listFile(imgPath):
        deleteFolder(tmpPath)
        createPath(tmpPath)
    
        imgFile = getFileName(i)
        #img = loadImg(i)
        #H,W = getImgHW(img)
        name = imgFile[:imgFile.rfind('.')]
        maskImgFile = maskImgPath + '\\' + name + '_mask' + '.png'
        maskImg = loadImg(maskImgFile)
        
        fAnnot = getImgAnnotFile(annotPath,i)

        aug = ImageColorAug(i,tmpPath)
        aug.augmentAll(N=6)
        
        def copyNewFile(path):
            print('start to copy...')
            for f in listFile(path):
                name_ = getFileName(f)
                newName = name_[:name_.rfind('.')]
                newFile = imgPath + '\\' + newName + '.png'
                newMaskFile = maskImgPath + '\\' + newName + '_mask' + '.png'
                newAnnoFile = annotPath + '\\' + newName + '.txt'
                print(newFile,newMaskFile)
                copyFile(f,newFile)
                copyFile(fAnnot,newAnnoFile)
                writeImg(maskImg,newMaskFile)
                           
        copyNewFile(tmpPath)
             

def main():
    base = r'.\res\PennFudanPed\\'
    annotPath = base + 'Annotation'
    imgPath = base + 'PNGImages'
    maskImgPath = base + 'PedMasks'
    
    #colorAugmentation(imgPath,maskImgPath,annotPath)
    #handleMaskLabel(maskImgPath)
    #return 

    newImgPath = base + 'newImages'
    
    dstImgPath = newImgPath + r'\newMaskScaling\\'
    dstMaskImgPath = newImgPath + r'\newMaskScalingMask\\'
    #print(dstImgPath,dstMaskImgPath)
    generateMaskByScaling(imgPath,maskImgPath,dstImgPath,dstMaskImgPath,rStart=1.0,N=1)
        
    dstImgPath = newImgPath + r'\newMaskCropping\\'
    dstMaskImgPath = newImgPath + r'\newMaskCroppingMask\\'
    generateMaskByClipping(imgPath,maskImgPath,annotPath, dstImgPath,dstMaskImgPath, N=4)

    dstImgPath = newImgPath + r'\newMaskFlipping\\'
    dstMaskImgPath = newImgPath + r'\newMaskFlippingMask\\'
    generateMaskByFlipping(imgPath,maskImgPath,dstImgPath,dstMaskImgPath)
        
if __name__ == '__main__':
    main()