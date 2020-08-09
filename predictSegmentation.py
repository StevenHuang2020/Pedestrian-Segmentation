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
    
    return parser.parse_args()

def preditImg(img, modelName = r'.\weights\trainPedSegmentation.h5'): 
    model = loadModel(modelName) #ks.models.load_model(modelName,custom_objects={'dice_coef_loss': dice_coef_loss})
    #model.summary()
    print(img.shape)
    
    x = img.reshape((-1,img.shape[0],img.shape[1],1))
    print(x.shape)
    mask = model.predict(x)
    
    print('preditImg mask.shape=',type(mask),mask.shape)
    return mask[0]

def demonstratePrediction(file):
    img = loadImg(file)
    img = resizeImg(img,256,256)
    maskImg = preditImg(img)
    bmaskImg = np.where(maskImg>0.5,1,0)
    
    c = np.unique(maskImg)
    print(c)

    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    ls.append(maskImg),nameList.append('maskImg')
    ls.append(bmaskImg),nameList.append('bmaskImg')

    plotImagList(ls, nameList,title='Segmentation prediction')
    
def comparePredict():
    img = loadGrayImg(r'.\res\PennFudanPed\newImages\trainImages\trainPNGImage\FudanPed00001_scale_0.2.png')
    gtMaskImg = loadGrayImg(r'.\res\PennFudanPed\newImages\trainImages\trainPNGImageMask\FudanPed00001_scale_0.2_mask.png')
    infoImg(img)
    infoImg(gtMaskImg)

    predMaskImg = preditImg(img)
    predMaskImg = np.where(predMaskImg>0.5,1,0).reshape((256,256))
    
    print('groundTrue=',gtMaskImg.shape, np.unique(gtMaskImg), np.sum(gtMaskImg))
    print('predMaskImg=',predMaskImg.shape, np.unique(predMaskImg), np.sum(predMaskImg))
    
    comparison = np.where(gtMaskImg[np.where(gtMaskImg==1)] == predMaskImg)
    #comparison = np.where(gtMaskImg==1 and gtMaskImg == predMaskImg)
    print(comparison)
    print('eqaul=',len(comparison[0]))
    
def main():
    arg = argCmdParse()    
    file=arg.source 
    dstFile = arg.dst
    print(file,dstFile)
    
    #demonstratePrediction(file)
    comparePredict()
    
if __name__=='__main__':
    #main()
    