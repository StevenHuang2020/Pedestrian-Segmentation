#python3
#Steven Image base operation Class
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

def changeBgr2Rbg(img): #input color img
    if getImagChannel(img) == 3:
        b,g,r = cv2.split(img)       # get b,g,r
        img = cv2.merge([r,g,b])
    return img

def changeRbg2Bgr(img): #when save
    if getImagChannel(img) == 3:
        r,g,b = cv2.split(img)       
        img = cv2.merge([b,g,r])
    return img

def loadImg(file, mode=cv2.IMREAD_COLOR, toRgb=True):
    #mode = cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE cv2.IMREAD_UNCHANGED
    try:
        img = cv2.imread(file,mode)
    except:
        print("Load image error,file=",file)
    
    assert(img is not None)
    if toRgb:
        if getImagChannel(img) == 3:
            img = changeBgr2Rbg(img)
    return img

def loadGrayImg(file):
    img = loadImg(file,mode=cv2.IMREAD_GRAYSCALE)
    return img

def writeImg(img,filePath):
    cv2.imwrite(filePath,img)

def infoImg(img,str='image:'):
    return print(str,'shape:',img.shape,'size:',img.size,'dims=',img.ndim,'dtype:',img.dtype)

def grayImg(img):
    if getImagChannel(img) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def getImagChannel(img):
    if img.ndim == 3: #color r g b channel
        return 3
    return 1  #only one channel

def resizeImg(img,NewW,NewH):
    try:
        return cv2.resize(img, (NewW,NewH), interpolation=cv2.INTER_CUBIC) #INTER_CUBIC INTER_NEAREST INTER_LINEAR INTER_AREA
    except:
        print('img.shape,newW,newH',img.shape,NewW,NewH)
        
def showimage(img,str='image',autoSize=False):
    flag = cv2.WINDOW_NORMAL
    if autoSize:
        flag = cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(str, flag)
    cv2.imshow(str,img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def plotImg(img,gray=False):
    if gray:
        plt.imshow(img,cmap="gray")
    else:
        plt.imshow(img)
            
    #plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
    plt.show()

def getImgHW(img):
    return img.shape[0],img.shape[1]

def distanceImg(img1,img2):
    return np.sqrt(np.sum(np.square(img1 - img2)))

"""-----------------------operation start-------"""
def calcAndDrawHist(img,color=[255,255,255]): #color histgram
    hist= cv2.calcHist([img], [0], None, [256], [0.0,255.0])
    #print(hist)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256,256,3], np.uint8) #[256,256,3] [256,256]
    hpt = int(0.9* 256)

    for h in range(256):
        intensity = int(hist[h]*hpt/maxVal)
        #print(h,intensity)
        cv2.line(histImg,(h,256), (h,256-intensity), color)
    return histImg

def equalizedHist(img):
    chn = getImagChannel(img)
    if chn == 1:
        return cv2.equalizeHist(img)
    else:
        newImg = img.copy()
        for n in range(chn):
            newImg[:,:,n] = cv2.equalizeHist(newImg[:,:,n])
        return newImg

def custEqualizedHist(img):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    
    newImage = np.zeros_like(img)
    if chn == 1:
        for n in range(chn):
            hist = cv2.calcHist([img],[n],None,[256],[0,256])
            cumSumP = np.cumsum(hist/(H*W))
                    
            for i in range(H):
                for j in range(W):
                    newImage[i,j] = 255*cumSumP[img[i,j]]
    else:        
        for n in range(chn):
            hist = cv2.calcHist([img],[n],None,[256],[0,256])
            #print(hist)
            p = hist/(H*W)
            #print(p)
            cumSumP = np.cumsum(p)
                    
            for i in range(H):
                for j in range(W):
                    newImage[i,j,n] = 255*cumSumP[img[i,j,n]]
    return newImage    
    
def histogram_equalize(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))

def binaryImage(img,thresH):
    """img must be gray"""
    H, W = getImgHW(img)
    newImage = img.copy()
    
    if 1:
        newImage = np.where(newImage<thresH,0,255)
    else:
        for i in range(H):
            for j in range(W):
                if newImage[i,j] < thresH:
                    newImage[i,j] = 0
                else:
                    newImage[i,j] = 255

    return newImage

def binaryImage2(img,thresHMin=0,thresHMax=0):
    """img must be gray"""
    H, W = getImgHW(img)
    newImage = img.copy()
    for i in range(H):
        for j in range(W):
            #print(newImage[i,j])
            if newImage[i,j] < thresHMin:
                newImage[i,j] = 0
            if newImage[i,j] > thresHMax:
                newImage[i,j] = 255

    return newImage

def thresHoldImage(img,thres=127,mode=cv2.THRESH_BINARY):
    #mode = cv2.THRESH_BINARY cv2.THRESH_BINARY_INV
    #cv2.THRESH_TRUNC cv2.THRESH_TOZERO_INV
    _, threshold = cv2.threshold(img,thres,255,mode)
    return threshold

def OtsuMethodThresHold(img):
    _, threshold = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    #blur = cv2.GaussianBlur(img,(5,5),0)
    #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshold
    
def thresHoldModel(img,mode=cv2.ADAPTIVE_THRESH_MEAN_C): 
    #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    return cv2.adaptiveThreshold(img,255,mode,cv2.THRESH_BINARY,11,2)

def convolutionImg(img,kernel):
    return cv2.filter2D(img,-1,kernel)
    
def colorSpace(img):
    #flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    #print(flags)
    #flag = cv2.COLOR_BGR2GRAY
    flag = cv2.COLOR_BGR2HSV
    return cv2.cvtColor(img, flag)

def flipImg(img,leftRight=True):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    
    newImage = img.copy()
    if leftRight:
        for j in range(W):
            if chn>1:
                for n in range(chn):
                    newImage[:, j, n] = img[:, W-j-1, n]
            else:
                newImage[:, j] = img[:, W-j-1]
    else:
        for i in range(H):
            if chn>1:
                for n in range(chn):
                    newImage[i, :, n] = img[H-i-1, :,n]
            else:
                newImage[i, :] = img[H-i-1, :]

    return newImage

def custGray(img,mean=True):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = np.zeros((H,W))
    if chn==1:
        return img
    
    if mean:    
        #grayscale = (R + G + B)/3 
        #newImage[:, :] = img[:, :, 0]
        newImage[:, :] = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])/3
    else:            
        #grayscale = ( (0.229 * R) + (0.587 * G) + (0.114 * B) )
        newImage[:, :] = (img[:, :, 0]*0.229 + img[:, :, 1]*0.587 + img[:, :, 2]*0.114)

    return newImage

def differImg(img, flag=0):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = np.zeros_like(img)
    
    if flag == 0: #leftRight
        for j in range(W-1):
            if chn>1:
                for n in range(chn):
                    newImage[:, j, n] = img[:, j+1, n] - img[:, j, n]
            else:
                newImage[:, j] = img[:, j+1] - img[:, j]
    elif flag == 1: #up down deviation
        for i in range(H-1):
            if chn>1:
                for n in range(chn):
                    newImage[i, :, n] = img[i+1, :, n] - img[i, :, n]
            else:
                newImage[i, :] = img[i+1, :] - img[i, :]
    else:
        for i in range(H-1):
            for j in range(W-1):
                if chn>1:
                    for n in range(chn):
                        a = (img[i, j+1, n] - img[i, j, n])**2
                        b = (img[i+1, j, n] - img[i, j, n])**2                        
                        newImage[i, j, n] = np.sqrt(a+b) #deviations
                else:
                    a = (img[i, j+1] - img[i, j])**2
                    b = (img[i+1, j] - img[i, j])**2                        
                    newImage[i, j] = np.sqrt(a+b) #deviations
                
    return newImage

def reverseImg(img):
    H,W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = np.zeros_like(img)
    #print(H,W)
    if chn ==1:
        for i in range(H):
            for j in range(W):
                newImage[i,j] = 255-img[i,j]
    else:
        for i in range(H):
            for j in range(W):
                newImage[i,j] = [255-img[i,j,0], 255-img[i,j,1], 255-img[i,j,2]]
    return newImage

def LightImg(img,ratio=1):
    H, W = getImgHW(img)
    chn =getImagChannel(img)
    newImage = img.copy()
    '''
    for i in range(H):
        for j in range(W):
            b = img[i,j,0]*ratio
            g = img[i,j,1]*ratio
            r = img[i,j,2]*ratio
            if b>255:
                b = 255
            if r>255:
                r = 255
            if g>255:
                g = 255

            newImage[i,j]=[b,g,r]
    '''
    '''
    b = img[:,:,0]*ratio
    g = img[:,:,1]*ratio
    r = img[:,:,2]*ratio
    b[b>255] = 255
    r[r>255] = 255
    g[g>255] = 255

    newImage[:,:,0]=b
    newImage[:,:,1]=g
    newImage[:,:,2]=r
    '''
    a = img[:,:,:]*ratio
    a[a>255] = 255
    newImage[:,:,:]=a
    return newImage

def contrastFilterImg(img):#filter fuction to extent [min,max]
    min = np.min(img)
    max = np.max(img)
    
    print('Before contrast:min,max=',min,max)
    
    def f(x):
        #return 100
        #return 1.5*x + 100
        #return f1(x)
        return f3(x)
    
    def f3(x):#Nonlinear Stretching  
        return 10*np.log(x+1) #c*log(x + 1)  c=1
        return np.power(x,2) # c*x^r  c=1,r=0.6
    
    def f2(x): #piecewise (0,0) (100,50) (200,150) (255,255)
        if x<100:
            return 50*x/100
        elif x<200:
            return x-50
        else:
            return 21*x/11 - 200*21/11 + 150
        
    def f1(x): #linear
        v = 255*(x-min)/(max-min)
        if v>255: v = 255
        return v
    
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = np.zeros_like(img)
    
    if chn == 1:
        for i in range(H):
            for j in range(W):
                newImage[i,j] = f(img[i,j])
    else:  
        for i in range(H):
            for j in range(W):
                b = img[i,j,0]
                g = img[i,j,1]
                r = img[i,j,2]
                newImage[i,j] = [f(b),f(g),f(r)]

    print('After contrast:min,max=',np.min(newImage),np.max(newImage))
    return newImage

def BaoguangImg(img, thres=128):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = np.zeros_like(img)
    
    def getBGOne(v):
        if v<thres:
            return 255-v
        return v
    
    if chn == 1:
        for i in range(H):
            for j in range(W):
                newImage[i,j] = getBGOne(img[i,j])
    else:
        for i in range(H):
            for j in range(W):
                newImage[i,j] = [getBGOne(img[i,j,0]),getBGOne(img[i,j,1]),getBGOne(img[i,j,2])]
    return newImage

def KuosanImg(img,N=3): #NxN jishu
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = img.copy()
    off = N//2
    for i in range(off,H-off):
        for j in range(off,W-off):
            lst = [img[i-1,j-1],img[i-1,j],img[i-1,j+1],img[i,j-1],img[i,j+1],img[i+1,j-1],img[i+1,j],img[i+1,j+1]]
            #id = random.randint(0,len(lst)-1)
            newImage[i,j] = random.choice(lst) #lst[id]
    return newImage

def meanImg(img):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    #mean = np.sum(img[:,:,:])/(H*W*chn)
    #print('mean = ',mean)
    return np.mean(img)

def varianceImg(img):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    mean = meanImg(img)
    if 1:
        return np.sqrt(np.sum((img-mean)**2)/(H*W*chn))
    else:
        variance = np.sum(img**2)/(H*W*chn)
        return variance - mean**2
    
def pyramidImg(img): #2x2-->1
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = np.zeros((H//2, W//2,chn),dtype=np.uint8)
    
    filter=[2,2]
    if chn == 1:
        for i in range(H//2):
            for j in range(W//2):
                startX=j*filter[1]
                stopX = startX + filter[1]
                startY=i*filter[0]
                stopY = startY + filter[0]
                
                newImage[i,j] = np.mean(img[startY:stopY,startX:stopX])
    else:
        for i in range(H//2):
            for j in range(W//2):
                startX=j*filter[1]
                stopX = startX + filter[1]
                startY=i*filter[0]
                stopY = startY + filter[0]
                
                r = np.mean(img[startY:stopY,startX:stopX,0])
                g = np.mean(img[startY:stopY,startX:stopX,1])
                b = np.mean(img[startY:stopY,startX:stopX,2])
                newImage[i,j] = [r,g,b]
       
    return newImage

def noiseImg(img,N):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    newImg = img.copy()
    
    #rdColor = np.random.randint(256, size=(1,1,chn))
    #print(rd)
    rdPointsX = np.random.randint(W, size=(1,N))
    #print('rdPointsX=',rdPointsX)
    rdPointsY = np.random.randint(H, size=(1,N))
    #print('rdPointsY=',rdPointsY)
    pts = np.hstack((rdPointsX.T, rdPointsY.T)) #np.concatenate((rdPointsX.T,rdPointsY.T), axis=0)
    #print('pts=',pts)
    
    for pt in pts:
        newImg[pt[0],pt[1],:] = np.random.randint(256, size=(1,1,chn))
    return newImg

def grayProjection(img):
    H, W = getImgHW(img)

    xPixNums = np.zeros((W), np.uint8)
    yPixNums = np.zeros((H), np.uint8)
    for i in range(H):#horizontal
        for j in range(W):
            if img[i,j] !=255:
                yPixNums[i]+=1
                
    for j in range(W): #vertical
        for i in range(H):
            if img[i,j] !=255:
                xPixNums[j]+=1

    return xPixNums,yPixNums

def autoThresholdValue(img,startMean=True):
    def newTValue(hist,T):
        value1 = 0
        num1 = 0
        
        value2 = 0
        num2 = 0
        for i,V in enumerate(hist): #2 class average gray value
            if i<=T:
                value1 = value1 + i*V
                num1 = num1 + V
            else:
                value2 = value2 + i*V
                num2 = num2 + V
            
        if num1 == 0:
            T1 = T
        else:
            T1 = value1/num1

        if num2 == 0:
            T2 = T
        else:
            T2 = value2/num2

        #print(T,T1,T2,'Next:',(T1+T2)/2)
        return (T1+T2)/2,T1,T2

    hist= cv2.calcHist([img], [0], None, [256], [0.0,255.0])
    hist = hist.ravel()
    #print('hist=',len(hist),hist)
    
    T = 0
    if startMean:
        T = cv2.mean(img)[0]
    print('T0=',T)
    TList=[]
    T1List=[]
    T2List=[]
    while(True):
        
        nT,T1,T2 = newTValue(hist,T)
        TList.append(T)
        T1List.append(T1)
        T2List.append(T2)
        if abs(nT-T)<0.2:
            break
        else:
            T = nT
    return T,TList,T1List,T2List

#https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
def cannyImg(img,threshold1=100,threshold2=200):
    return cv2.Canny(img, threshold1=threshold1, threshold2=threshold2)

def sobelImg(img,scale=1,delta=0,ddepth = cv2.CV_16S, ksize=3):
    return cv2.Sobel(img, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
 
def textImg(img,str,loc=None,color=(0,0,0),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1,thickness = 1):
    H,W = getImgHW(img)
    newImg = img.copy()
    textSize = cv2.getTextSize(str, fontFace, fontScale, thickness)
    print('textSize=',textSize)
    if loc is None:
        loc = ((W - textSize[0][0])//2, (H + textSize[0][1])//2)
    return cv2.putText(newImg, str,loc, fontFace, fontScale, color, thickness, cv2.LINE_AA)

def jointImage(img1,img2,hori=True):
    H1,W1 = getImgHW(img1)
    H2,W2 = getImgHW(img2)
    if hori:
        H = np.max(H1,H2)
        img = np.zeros((H, W1+W2, 3),dtype=np.uint8)
        img[:H1, 0:W1] = img1
        img[:H2, W1:] = img2
    else:
        W = np.max(W1,W2)
        img = np.zeros((H1+H2, W, 3),dtype=np.uint8)
        img[:H1, 0:W1] = img1
        img[H1:, 0:W2] = img2
    return img

def rectangleImg(img,startPt,stopPt,color=(0, 0, 255),thickness=2):
    #return cv2.rectangle(img=img, pt1=startPt, pt2=stopPt, color=color, thickness=thickness) 
    return cv2.rectangle(img, startPt, stopPt, color, thickness=2)

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


def blurImg(img,ksize=5): #Averaging adjency pixsel 5x5 size kernel
    return cv2.blur(img, (ksize,ksize))

def gaussianBlurImg(img, ksize=5): #Gaussian Blurring
    kernel = cv2.getGaussianKernel(ksize,0)
    #print('gaussian kernel=',kernel)
    return cv2.GaussianBlur(img,(ksize,ksize),0)

def medianBlurImg(img,ksize=5): #Median Blurring
    return cv2.medianBlur(img, ksize)

def bilateralBlurImg(img,d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(img,d,sigmaColor,sigmaSpace)