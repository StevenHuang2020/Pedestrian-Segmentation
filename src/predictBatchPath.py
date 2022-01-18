# python3 steven

# import os
# import sys
# import cv2
import argparse

# ----------------------------------------------
# usgae: python .\predictBatchPath.py
# ----------------------------------------------
from predictSegmentation import getPredictionMaskImg
from modules.folder.folder_file import pathsFiles, createPath, getFileName
from commonModule.ImageBase import loadImg, writeImg


def cmd_line():
    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--src', required=True, help='path to input image')
    ap.add_argument('-d', '--dst', required=True, help='path to save image')
    return ap.parse_args()


def main():
    args = cmd_line()
    src = args.src
    dst = args.dst

    createPath(dst)
    print('src=', src, 'dst=', dst)

    for i in pathsFiles(src, 'jpg'):  # png
        fileName = getFileName(i)
        img = loadImg(i)

        dstFile = args.dst + '\\' + fileName
        print(i, fileName, dstFile)
        predImg = getPredictionMaskImg(img)
        writeImg(predImg, dstFile)


if __name__ == '__main__':
    main()
