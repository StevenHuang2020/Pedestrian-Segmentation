import os,sys
sys.path.append('..')
#print(sys.path)

import tensorflow.keras as ks
import numpy as np
import argparse
from ImageBase import *
from mainImagePlot import plotImagList
#--------------------------------------------------------------------------------------
#usgae: python predictSegmentation.py -s .\res\PennFudanPed\PNGImages\FudanPed00001.png
#--------------------------------------------------------------------------------------

def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help = 'source image')
    parser.add_argument('-d', '--dst', help = 'save iamge')
    
    return parser.parse_args()

def preditImg(img, modelName = r'.\weights\old\trainPedSegmentation.h5'): 
    model = ks.models.load_model(modelName)
    print(img.shape)
    
    x = img.reshape((-1,img.shape[0],img.shape[1],3))
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
    img = loadImg(r'.\res\PennFudanPed\newImages\trainImages\trainPNGImages\FudanPed00001_scale_0.2.png')
    img = resizeImg(img,256,256)
    gtMaskImg = loadImg(r'.\res\PennFudanPed\newImages\trainImages\trainPNGImageMask\FudanPed00001_scale_0.2_mask.png')
    gtMaskImg = resizeImg(gtMaskImg,256,256)
    gtMaskImg = gtMaskImg[:,:,0]
    
    predMaskImg = preditImg(img)
    predMaskImg = np.where(predMaskImg>0.5,1,0)
    
    print('groundTrue=',gtMaskImg.shape, np.unique(gtMaskImg), np.sum(gtMaskImg))
    print('predMaskImg=',predMaskImg.shape, np.unique(predMaskImg), np.sum(predMaskImg))
    
    print('eqaul=',np.sum(np.where(gtMaskImg == 1 and gtMaskImg==predMaskImg)))
    
def main():
    arg = argCmdParse()    
    file=arg.source 
    dstFile = arg.dst
    print(file,dstFile)
    
    #demonstratePrediction(file)
    comparePredict()
    
if __name__=='__main__':
    main()