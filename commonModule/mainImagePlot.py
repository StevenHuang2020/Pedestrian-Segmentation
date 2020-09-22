#python3
#Steven 11/03/2020 image plot modoule
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
from common import getRowAndColumn
from ImageBase import *

def plotImg2(img,name='', title='', gray=False, showticks=True):
    ls,nameList = [],[]
    ls.append(img),nameList.append(name)
    plotImagList(ls, nameList, title=title, gray=gray, showticks=showticks)
    
def plotImagList(imgList, nameList, title='', gray=False, showticks=True):
    nImg = len(imgList)
    nRow,nColumn = getRowAndColumn(nImg)
    
    plt.figure().suptitle(title, fontsize="x-large")
    for n in range(nImg):
        img = imgList[n]
        ax = plt.subplot(nRow, nColumn, n + 1)
        ax.title.set_text(nameList[n])
        if gray:
            plt.imshow(img,cmap="gray")
        else:
            plt.imshow(img)
        
        if not showticks:
            ax.set_yticks([])
            ax.set_xticks([])
    #plt.grid(True)
    plt.tight_layout()
    
    # plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    # plt.margins(0,0)
    # plt.savefig('result.png', dpi=300)
    #plt.savefig('result.png',bbox_inches='tight',dpi=300,pad_inches=0.0)
    plt.show()

def main():
    file = r'./res/obama.jpg'#'./res/Lenna.png' #
    img = loadImg(file,mode=cv2.IMREAD_GRAYSCALE) # IMREAD_GRAYSCALE IMREAD_COLOR
    infoImg(img)
    img = binaryImage2(img,thresHMin=50,thresHMax=150)
    showimage(img)
    pass

if __name__=='__main__':
    main()
