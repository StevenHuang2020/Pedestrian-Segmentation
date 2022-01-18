# python3
# Steven Image color augmentation Class
import cv2
import os
import numpy as np
# import random
from commonModule.ImageBase import loadImg, blurImg, writeImg
from commonModule.ImageBase import gaussianBlurImg, medianBlurImg, adjustBrightnessAndContrast, GammaCorrection


class ImageColorAug():
    methods = ['blur', 'rgb channel', 'brightness', 'gamma correction']

    def __init__(self, file, dstImgPath):
        self.image = loadImg(file)
        self.fileName = os.path.basename(file)
        self.imgName = self.fileName[:self.fileName.rfind('.')]
        print('ImageColorAug image:', file, self.fileName, self.imgName)
        assert(self.image is not None)
        self.dstImgPath = self._createPath(dstImgPath)

    def _createPath(self, dirs):
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        return dirs

    def augmentAll(self, N):
        for m in ImageColorAug.methods:
            self.augmentByMethod(m, N)

    def augmentByMethod(self, method, N):
        if method == ImageColorAug.methods[0]:
            self._augmentByBlur(N, blurImg, 'blurImg')
            self._augmentByBlur(N, gaussianBlurImg, 'gaussianBlurImg')
            self._augmentByBlur(N, medianBlurImg, 'medianBlurImg')
        elif method == ImageColorAug.methods[1]:
            self._augmentRgbChn()
        elif method == ImageColorAug.methods[2]:
            self._augmentByBrightnessAndContrast(N)
        elif method == ImageColorAug.methods[3]:
            self._augmentByGammaCorrection(N)

    def _augmentByBlur(self, numbers, fun, funName):
        for i in range(numbers):
            size = i + 2
            if funName == 'gaussianBlurImg' or funName == 'medianBlurImg':
                if size % 2 == 0:
                    continue
            img = fun(self.image.copy(), ksize=size)
            newName = r'/' + self.imgName + '_' + funName + '_s' + str(size) + '.png'
            print('newName=', newName)
            writeImg(img, self.dstImgPath + newName)

    def _augmentRgbChn(self):
        chns = ['b', 'g', 'r']
        for i, img in enumerate(cv2.split(self.image)):
            newName = r'/' + self.imgName + '_' + chns[i] + '.png'
            print('newName=', newName)
            writeImg(img, self.dstImgPath + newName)

    def _augmentByBrightnessAndContrast(self, numbers):
        beta = 50
        for alpha in np.linspace(0.2, 2, numbers):
            newName = r'/' + self.imgName + 'alphaBeta' + '_' + str(alpha) + '_' + str(beta) + '.png'
            print('newName=', newName)
            img = adjustBrightnessAndContrast(self.image.copy(), alpha, beta)
            writeImg(img, self.dstImgPath + newName)

        alpha = 0.6
        for beta in np.linspace(10, 100, numbers):
            newName = r'/' + self.imgName + 'alphaBeta' + '_' + str(alpha) + '_' + str(beta) + '.png'
            print('newName=', newName)
            img = adjustBrightnessAndContrast(self.image.copy(), alpha, beta)
            writeImg(img, self.dstImgPath + newName)

    def _augmentByGammaCorrection(self, numbers):
        for gamma in np.linspace(0.1, 5, numbers):
            # gamma = gamma.round(2)
            newName = r'/' + self.imgName + 'gamma' + '_' + str(gamma) + '.png'
            print('newName=', newName)
            writeImg(GammaCorrection(self.image.copy(), gamma), self.dstImgPath + newName)


if __name__ == '__main__':
    file = r'./res/Lenna.png'
    dstPath = r'./res/colorAug'
    aug = ImageColorAug(file, dstPath)
    aug.augmentAll(N=10)
