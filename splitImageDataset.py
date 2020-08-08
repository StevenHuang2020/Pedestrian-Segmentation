#python3 steven
import sys
from modules.folder.folder_file import createPath,getFileName,pathsFilesList
from modules.folder.folder_file import deleteFolder,copyFile,deleteFile
from genImageBoxLabel import writeToDst

def train_test_split(srcPath,maskImgPath,dstTrainPath,dstTestPath,dstTrainMaskPath,dstTestMaskPath,test_size=0.3):
    files = pathsFilesList(srcPath,filter='png')
    total = len(files)
    trainLen = int(total*(1-test_size))
    print('total,trainLen,testLen=',total,trainLen,total-trainLen)
    for i,f in enumerate(files):
        fileName = getFileName(f)
        maskFileName = fileName[:fileName.rfind('.')] + '_mask.png'
        
        fMaskFile = maskImgPath + '\\' + maskFileName
        print('fMaskFile=',fMaskFile)
        if i<trainLen:
            copyFile(f,dstTrainPath + '\\' + fileName)
            copyFile(fMaskFile,dstTrainMaskPath + '\\' + maskFileName)
        else:
            copyFile(f,dstTestPath + '\\' + fileName)
            copyFile(fMaskFile,dstTestMaskPath + '\\' + maskFileName)

        
def collectImageDataset(srcPath,maskImgPath,dstImgPath,dstMaskImgPath,fileListFile):
    files = pathsFilesList(srcPath,filter='png')
    total = len(files)
   
    for i,f in enumerate(files):
        fileName = getFileName(f)
        maskFileName = fileName[:fileName.rfind('.')] + '_mask.png'
        
        fMaskFile = maskImgPath + '\\' + maskFileName
        print('fMaskFile=',fMaskFile)
        
        dstImg = dstImgPath + '\\' + fileName
        dstMaskImg = dstMaskImgPath + '\\' + maskFileName
        copyFile(f, dstImg)
        copyFile(fMaskFile, dstMaskImg)
       
        writeToDst(fileListFile, dstImg + ',' + dstMaskImg + '\n') #new  trainList.list
        
def main():    
    base = r'.\res\PennFudanPed\newImages\trainImages\\'
    
    dstImgPath = base + r'trainPNGImages'
    dstMaskImgPath = base + r'trainPNGImageMask'
    fileListFile =  base + r'trainList.list'
    
    deleteFile(fileListFile)
    deleteFolder(dstImgPath)
    createPath(dstImgPath)
    deleteFolder(dstMaskImgPath)
    createPath(dstMaskImgPath)
    
    '''
    dstTrainPath = base + r'train_PNGImages'
    dstTestPath = base + r'test_PNGImages'
    dstTrainMaskPath =  base + r'train_MaskImages'
    dstTestMaskPath =  base + r'test_MaskImages'
    
    
    deleteFolder(dstTrainPath)
    createPath(dstTrainPath)
    deleteFolder(dstTestPath)
    createPath(dstTestPath)
    deleteFolder(dstTrainMaskPath)
    createPath(dstTrainMaskPath)
    deleteFolder(dstTestMaskPath)
    createPath(dstTestMaskPath)
    '''
    imgPathList=[]
    imgPathList.append((r'.\res\PennFudanPed\newImages\newMaskCropping',r'.\res\PennFudanPed\newImages\newMaskCroppingMask'))
    imgPathList.append((r'.\res\PennFudanPed\newImages\newMaskFlipping',r'.\res\PennFudanPed\newImages\newMaskFlippingMask'))
    imgPathList.append((r'.\res\PennFudanPed\newImages\newMaskScaling',r'.\res\PennFudanPed\newImages\newMaskScalingMask'))
    
    for i in imgPathList:
        imgPath = i[0]
        maskImgPath = i[1]
        #train_test_split(imgPath,maskImgPath,dstTrainPath,dstTestPath,dstTrainMaskPath,dstTestMaskPath,test_size=0.2)
        collectImageDataset(imgPath,maskImgPath,dstImgPath,dstMaskImgPath,fileListFile)
    
if __name__=='__main__':
    main()
    