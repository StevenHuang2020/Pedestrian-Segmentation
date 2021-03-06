#python3 steven
import cv2
import argparse
import numpy as np
import os

def pathsFiles(dir,filter='',subFolder=False): #"cpp h txt jpg"
    def getExtFile(file):
        return file[file.find('.')+1:]
    
    def getFmtFile(path):
        #/home/User/Desktop/file.txt    /home/User/Desktop/file     .txt
        root_ext = os.path.splitext(path) 
        return root_ext[1]

    fmts = filter.split()    
    if fmts:
        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                if getExtFile(getFmtFile(filename)) in fmts:
                    yield dirpath+'\\'+filename
            if not subFolder:
                break
    else:
        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                yield dirpath+'\\'+filename  
            if not subFolder:
                break    
            
def createPath(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        
def getFileName(path):  
    return os.path.basename(path)
    
def jointImage(img1,img2,hori=True):
    assert(img1.shape == img2.shape)
    H,W = img1.shape[0],img1.shape[1]
    if hori:
        img = np.zeros((H, W*2, 3),dtype=np.uint8)
        img[:, 0:W] = img1
        img[:, W:] = img2
    else:
        img = np.zeros((H*2, W, 3),dtype=np.uint8)
        img[0:H, :] = img1
        img[H:, :] = img2
    return img

'''
def loadImg(file):
    return cv2.imread(file, flags=cv2.IMREAD_COLOR) #skimage.io.imread(file)

def writeImg(file,img):
    cv2.imwrite(file, img)
    
def resizeImg(img,NewW,NewH):
    return cv2.resize(img, (NewW,NewH), interpolation=cv2.INTER_CUBIC) #INTER_CUBIC INTER_NEAREST INTER_LINEAR INTER_AREA
'''

def JointImagePath():
    src1=r'E:\python\AI\yolo\darknet-master\video\png'
    src2=r'E:\python\AI\yolo\darknet-master\video\png\segdst'
    dst=r'E:\python\AI\yolo\darknet-master\video\png\segDstVideo'
    
    createPath(dst)
    print(src1,src2)
    for i in pathsFiles(src1,'png'): #png
        fileName = getFileName(i)
        img1 = loadImg(i)
        j = src2 + '\\' + fileName
        img2 = loadImg(j)
        assert(img1 is not None and img2 is not None)
        
        img1 = resizeImg(img1,img1.shape[1]//2,img1.shape[0]//2)
        img2 = resizeImg(img2,img2.shape[1]//2,img2.shape[0]//2)
        dstImg = jointImage(img1,img2)
        
        dstFile = dst + '\\' + fileName
        print('start to joint:',i,j) #dstFile
        writeImg(dstFile,dstImg)
        #break
    
def main():
    JointImagePath()

if __name__ == '__main__':
    main()
