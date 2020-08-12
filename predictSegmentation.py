import os,sys
sys.path.append('..')
#print(sys.path)

import tensorflow.keras as ks
import numpy as np
import argparse
from ImageBase import *
from mainImagePlot import plotImagList
from mainTrainning import loadModel
#--------------------------------------------------------------------------------------
#usgae: python predictSegmentation.py -s .\res\PennFudanPed\PNGImages\FudanPed00001.png
#--------------------------------------------------------------------------------------

def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help = 'source image')
    parser.add_argument('-d', '--dst', help = 'save iamge')
    parser.add_argument('-c', dest='compare', action='store_true')
    
    return parser.parse_args()

def preditImg(img, modelName = r'.\weights\trainPedSegmentation.h5'): 
    model = loadModel(modelName) #ks.models.load_model(modelName,custom_objects={'dice_coef_loss': dice_coef_loss})
    #model.summary()
    print(img.shape)
    
    x = img.reshape((-1,img.shape[0],img.shape[1],1))
    print(x.shape)
    mask = model.predict(x)
    
    print('preditImg mask.shape=',type(mask),mask.shape)
    predict = mask[0] 
    predict = predict.reshape((predict.shape[0],predict.shape[1]))
    return predict

def processMaskImg(img,backColor=0):
    H,W = getImgHW(img)
    chn = getImagChannel(img)
    cl = np.unique(img)
    print(cl,chn,H,W)
    #colors = np.random.uniform(0, 255, size=(len(cl), chn))
    #print('colors=', colors)
    colors = np.array([[255,0,0],[0,255,0],[0,0,255]])
    print('colors=', colors[0])
    newImg = np.zeros_like(img)
    for i in range(H):
        for j in range(W):
            #newImg[i,j,:] = backColor
            if img[i,j,0] == 0:
                newImg[i,j,:] = backColor
            else:
                newImg[i,j,:] = colors[img[i,j,0]]
    return newImg
    
def expandImageTo3chan(img):
    if getImagChannel(img) == 3:
        rimg = img
    else:
        img = img.reshape((img.shape[0],img.shape[1]))
        rimg = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
        rimg[:,:,0] = img
        rimg[:,:,1] = img
        rimg[:,:,2] = img
    return rimg

def maskToOrignimalImg(img,maskImg):
    mask = maskImg
    if img.shape != maskImg.shape:
        mask = expandImageTo3chan(maskImg)

    return cv2.addWeighted(img, 1.0, mask, 0.8, 0)

    H,W = getImgHW(img)
    chn = getImagChannel(img)
    #print(img.shape,maskImg.shape)
    cl = np.unique(maskImg)
    colors = np.random.uniform(0, 255, size=(len(cl), chn))
    newImg = img.copy()
    for i in range(H):
        for j in range(W):
            if maskImg[i,j,0] != 0:
                newImg[i,j,:] = colors[maskImg[i,j,0]]
            
    return newImg
    
def getPredictionMaskImg(img):
    predMaskImg = preditImg(grayImg(img))
    binPredMaskImg = np.where(predMaskImg>0.5,1,0)
    binPredMaskImg = binPredMaskImg.astype(np.uint8)
    
    binPredMaskImg = expandImageTo3chan(binPredMaskImg)
    binPredMaskImg = processMaskImg(binPredMaskImg)
    binPredMaskImg = binPredMaskImg.astype(np.uint8)
    return maskToOrignimalImg(img,binPredMaskImg)

def demonstratePrediction(file,gtMaskFile=None):
    img = loadImg(file)
    img = resizeImg(img,256,256)
    
    pImg = getPredictionMaskImg(img)
    
    if gtMaskFile is not None:
        maskImg = loadImg(gtMaskFile)
        maskImg = resizeImg(maskImg,256,256)
        maskImg = processMaskImg(maskImg)
        mImg = maskToOrignimalImg(img,maskImg)
    

    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    if gtMaskFile is not None:
        ls.append(maskImg),nameList.append('maskImg')
        ls.append(mImg),nameList.append('GT Img')
    ls.append(pImg),nameList.append('predict Img')

    plotImagList(ls, nameList,gray=True,title='Segmentation prediction',showticks=False)

def demonstrateGrayPrediction(file):
    img = loadGrayImg(file)
    img = resizeImg(img,256,256)
    maskImg = preditImg(img)
    bmaskImg = np.where(maskImg>0.5,1,0)
    
    c = np.unique(maskImg)
    print('c=',c)

    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    ls.append(maskImg),nameList.append('PredmaskImg')
    ls.append(bmaskImg),nameList.append('PredbmaskImg')

    plotImagList(ls, nameList,gray=True,title='Segmentation prediction')
    
def comparePredict():
    file = r'.\res\PennFudanPed\newImages\trainImages\trainPNGImage\FudanPed00001_scale_0.2.png'
    mask = r'.\res\PennFudanPed\newImages\trainImages\trainPNGImageMask\FudanPed00001_scale_0.2_mask.png'
    img = loadGrayImg(file)
    gtMaskImg = loadGrayImg(mask)
    infoImg(img)
    infoImg(gtMaskImg)

    predMaskImg = preditImg(img)
    predMaskImg = np.where(predMaskImg>0.5,1,0).reshape((256,256))
    
    print('groundTrue=',gtMaskImg.shape, np.unique(gtMaskImg), np.sum(gtMaskImg))
    print('predMaskImg=',predMaskImg.shape, np.unique(predMaskImg), np.sum(predMaskImg))
    
    #comparison = np.where(gtMaskImg[np.where(gtMaskImg==1)] == predMaskImg)
    #comparison = np.where(gtMaskImg==1 and gtMaskImg == predMaskImg)
    comparison = np.sum(gtMaskImg*predMaskImg)
    print('eqaul=',comparison)
    
def testCombing():
    file = r'.\res\PennFudanPed\PNGImages\FudanPed00001.png'
    maske = r'.\res\PennFudanPed\PedMasks\FudanPed00001_mask.png'
    
    background = loadImg(file)
    mask = loadImg(maske)
    mask = processMaskImg(mask,backColor=0)
    added_image = cv2.addWeighted(background, 1.0, mask, 0.8, 0)
    
    ls,nameList = [],[]
    ls.append(background),nameList.append('background')
    ls.append(mask),nameList.append('mask')
    ls.append(added_image),nameList.append('added_image')

    plotImagList(ls, nameList, gray=True, title='Combine')
    
def main():
    #return testCombing()
    arg = argCmdParse()    
    file=arg.source 
    dstFile = arg.dst
    print(file,dstFile)
    
    if arg.compare:
        comparePredict()
    else:
        #demonstrateGrayPrediction(file)
        file=r'.\res\PennFudanPed\newImages\trainImages\trainPNGImage\FudanPed00001_17_104_557_490.png'
        maskFile=r'.\res\PennFudanPed\newImages\trainImages\trainPNGImageMask\FudanPed00001_17_104_557_490_mask.png'
        file=r'.\res\PennFudanPed\PNGImages\FudanPed00012.png'
        maskFile=r'.\res\PennFudanPed\PedMasks\FudanPed00012_mask.png'
        
        #demonstratePrediction(file,maskFile)

        file=r'.\res\others\2.jpg'
        demonstratePrediction(file)
        
    
if __name__=='__main__':
    main()
    
    # a = np.array([[0,1,0],
    #               [0,1,1]])
    # b = np.array([[0,1,1],
    #              [0,1,1]])
    # comparison = a*b #np.where(a[np.where(a==1)] == b[np.where(b==1)])
    # print('c=',comparison,np.sum(comparison))