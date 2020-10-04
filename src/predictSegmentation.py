import os,sys
sys.path.append(r'./commonModule')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #close tf debug info

import numpy as np
import argparse
from commonModule.ImageBase import *
from commonModule.mainImagePlot import plotImagList
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

def preditImg(img, modelName = r'..\weights\trainPedSegmentation.h5'): 
    model = loadModel(modelName) #ks.models.load_model(modelName,custom_objects={'dice_coef_loss': dice_coef_loss})
    #model.summary()
    #print(img.shape)
    img = resizeImg(img,256,256)
    assert(img.shape[0]==256 and img.shape[1]==256)
    
    x = img.reshape((-1,img.shape[0],img.shape[1],1))
    #print(x.shape)
    mask = model.predict(x)
    
    print('preditImg mask.shape=',type(mask),mask.shape)
    predict = mask[0] 
    predict = predict.reshape((predict.shape[0],predict.shape[1]))
    
    predict = np.where(predict>0.5,1,0)
    predict = predict.astype(np.uint8)
    return predict

def processMaskImg(img,backColor=0):
    H,W = getImgHW(img)
    chn = getImagChannel(img)
    cl = np.unique(img)
    print(cl,chn,H,W)
    #colors = np.random.uniform(0, 255, size=(len(cl), chn))
    #print('colors=', colors)
    #colors = np.array([[255,0,0],[0,255,0],[0,0,255]])  #3 colors
    colors = np.array([[0,255,0],[0,255,0],[0,255,0]])  #one colors
    #print('colors=', colors[0])
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

def maskToOrignimalImg(img,mask):
    """add mask to color image"""
    if img.shape != mask.shape:
        #mask = expandImageTo3chan(mask)
        img = expandImageTo3chan(img)

    #print(img.shape,mask.shape)
    return cv2.addWeighted(img, 1.0, mask, 0.8, 0)

def getPredictionMaskImg(img):
    H,W = getImgHW(img)
    blurImg = gaussianBlurImg(img.copy(),3)
    
    predMaskImg = preditImg(grayImg(blurImg))
     
    predMaskImg = expandImageTo3chan(predMaskImg)
    predMaskImg = processMaskImg(predMaskImg)
    predMaskImg = predMaskImg.astype(np.uint8)
    return resizeImg(predMaskImg,W,H)

def getPredictImg(img):
    pMask = getPredictionMaskImg(img.copy())
    return pMask, maskToOrignimalImg(img, pMask)
  
def demonstratePrediction(file,gtMaskFile=None):
    img = loadImg(file)
    pMask,pImg = getPredictImg(img) #prediction
    
    if gtMaskFile is not None:
        maskImg = loadImg(gtMaskFile)
        #maskImg = resizeImg(maskImg,256,256)
        maskImg = processMaskImg(maskImg)
        mImg = maskToOrignimalImg(img,maskImg)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    if gtMaskFile is not None:
        ls.append(maskImg),nameList.append('GT mask')
        #ls.append(mImg),nameList.append('GT Segmentation')
    ls.append(pImg),nameList.append('Predict Segmentation')
    
    if gtMaskFile is not None:
        ls.append(pMask),nameList.append('Predict Mask')
    
    plotImagList(ls, nameList,gray=True,showticks=False) #title='Segmentation prediction',

def demonstrateGrayPrediction(file):
    img = loadGrayImg(file)
    _,pImg = getPredictImg(img)

    #c = np.unique(pImg)
    #print('c=',c)

    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    ls.append(pImg),nameList.append('PredmaskImg')
    plotImagList(ls, nameList,gray=True,title='Segmentation prediction')
    
def comparePredict():
    file = r'.\res\PennFudanPed\PNGImages\FudanPed00001.png'
    mask = r'.\res\PennFudanPed\PedMasks\FudanPed00001_mask.png'
    img = loadGrayImg(file)
    gtMaskImg = loadGrayImg(mask)
    gtMaskImg = resizeImg(gtMaskImg,256,256)
    #infoImg(img)
    #infoImg(gtMaskImg)

    predMaskImg = preditImg(img)
    
    print('groundTrue=',gtMaskImg.shape, np.unique(gtMaskImg), np.sum(gtMaskImg))
    print('predMaskImg=',predMaskImg.shape, np.unique(predMaskImg), np.sum(predMaskImg))
    
    #comparison = np.where(gtMaskImg[np.where(gtMaskImg==1)] == predMaskImg)
    #comparison = np.where(gtMaskImg==1 and gtMaskImg == predMaskImg)
    comparison = np.sum(gtMaskImg*predMaskImg)
    print('eqaul=',comparison)
    
def getImageMask(file,maskeFile):
    img = loadImg(file)
    mask = loadImg(maskeFile)
    mask = processMaskImg(mask,backColor=0)
    #added_image = cv2.addWeighted(background, 1.0, mask, 0.8, 0)
    return img,mask
    
def testCombing():
    file = r'.\res\PennFudanPed\PNGImages\FudanPed00001.png'
    maske = r'.\res\PennFudanPed\PedMasks\FudanPed00001_mask.png'
    crop_file = r'.\res\PennFudanPed\newImages\newMaskCropping\FudanPed00001_b_143_15_553_524.png'
    crop_mask = r'.\res\PennFudanPed\newImages\newMaskCroppingMask\FudanPed00001_b_143_15_553_524_mask.png'
    f_file=r'.\res\PennFudanPed\newImages\newMaskFlipping\FudanPed00001_flip.png'
    f_mask=r'.\res\PennFudanPed\newImages\newMaskFlippingMask\FudanPed00001_flip_mask.png'
    color_file = r'.\res\PennFudanPed\PNGImages\FudanPed00001alphaBeta_2.0_50.png'
    color_mask = r'.\res\PennFudanPed\PedMasks\FudanPed00001alphaBeta_2.0_50_mask.png'
    
    img,mask = getImageMask(file,maske)
    crop_img,crop_mask = getImageMask(crop_file,crop_mask)
    f_img,f_mask = getImageMask(f_file,f_mask)
    c_img,c_mask = getImageMask(color_file,color_mask)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('Orignial')
    ls.append(mask),nameList.append('mask')
    ls.append(crop_img),nameList.append('Cropping')
    ls.append(crop_mask),nameList.append('mask')
    ls.append(f_img),nameList.append('Flipping')
    ls.append(f_mask),nameList.append('mask')
    
    ls.append(c_img),nameList.append('Color change')
    ls.append(c_mask),nameList.append('mask')
    #ls.append(added_image),nameList.append('added_image')

    plotImagList(ls, nameList, gray=True, title='', showticks=False)
    
def main():
    arg = argCmdParse()    
    file=arg.source 
    dstFile = arg.dst
    print(file,dstFile)
    
    if arg.compare:
        comparePredict()
    else:       
        if 0: 
            # file=r'.\res\PennFudanPed\newImages\trainImages\trainPNGImage\FudanPed00001_17_104_557_490.png'
            # maskFile=r'.\res\PennFudanPed\newImages\trainImages\trainPNGImageMask\FudanPed00001_17_104_557_490_mask.png'
            file=r'.\res\PennFudanPed\PNGImages\FudanPed00012.png'
            maskFile=r'.\res\PennFudanPed\PedMasks\FudanPed00012_mask.png'
            
            file=r'.\res\PennFudanPed\PNGImages\PennPed00068.png'
            maskFile=r'.\res\PennFudanPed\PedMasks\PennPed00068_mask.png'
            
            demonstratePrediction(file,maskFile)
        else:
            #file=r'.\res\7.jpg'
            #demonstrateGrayPrediction(file)
            demonstratePrediction(file)
        
    
if __name__=='__main__':
    main()
    