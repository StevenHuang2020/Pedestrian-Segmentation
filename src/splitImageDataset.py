# python3 steven
from modules.folder.folder_file import createPath, getFileName, pathsFilesList
from modules.folder.folder_file import deleteFolder, copyFile, deleteFile
from genImageBoxLabel import writeToDst
from mainAugmentation import writeImg, resizeImg, loadImg, grayImg

# global
trainImgWidth = 256
trainImgHeight = 256


def collectImageDataset(srcPath, maskImgPath, dstImgPath, dstMaskImgPath, fileListFile):
    files = pathsFilesList(srcPath, filter='png')
    # total = len(files)

    for i, f in enumerate(files):
        fileName = getFileName(f)
        maskFileName = fileName[:fileName.rfind('.')] + '_mask.png'

        fMaskFile = maskImgPath + '\\' + maskFileName
        # print('fMaskFile=',fMaskFile)

        dstImg = dstImgPath + '\\' + fileName
        dstMaskImg = dstMaskImgPath + '\\' + maskFileName

        # copyFile(f, dstImg)
        # copyFile(fMaskFile, dstMaskImg)
        img = loadImg(f)
        img = grayImg(resizeImg(img, newH=trainImgHeight, newW=trainImgWidth))
        assert(img is not None)
        writeImg(img, dstImg)

        imgMask = loadImg(fMaskFile)
        imgMask = grayImg(resizeImg(imgMask, newH=trainImgHeight, newW=trainImgWidth))
        assert(imgMask is not None)
        writeImg(imgMask, dstMaskImg)

        # print('img.shape=',img.shape)
        # print('imgMask.shape=',imgMask.shape)
        writeToDst(fileListFile, dstImg + ',' + dstMaskImg + '\n')  # new  trainList.list


def main():
    base = r'.\res\PennFudanPed\newImages\trainImages\\'

    dstImgPath = base + r'trainPNGImage'
    dstMaskImgPath = base + r'trainPNGImageMask'
    fileListFile = base + r'trainList.list'

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
    imgPathList = []
    imgPathList.append((r'.\res\PennFudanPed\newImages\newMaskCropping', r'.\res\PennFudanPed\newImages\newMaskCroppingMask'))
    imgPathList.append((r'.\res\PennFudanPed\newImages\newMaskFlipping', r'.\res\PennFudanPed\newImages\newMaskFlippingMask'))
    imgPathList.append((r'.\res\PennFudanPed\newImages\newMaskScaling', r'.\res\PennFudanPed\newImages\newMaskScalingMask'))

    for i in imgPathList:
        imgPath = i[0]
        maskImgPath = i[1]
        collectImageDataset(imgPath, maskImgPath, dstImgPath, dstMaskImgPath, fileListFile)


if __name__ == '__main__':
    main()
