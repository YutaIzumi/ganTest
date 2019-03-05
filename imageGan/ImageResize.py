# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:32:20 2019

@author: izumiy
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:22:09 2019

@author: izumiy
"""

# 画像をリサイズして、numpy配列で返すモジュール

import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ファイル名の一括取得
def getAllFileName():
    names = glob.glob("./img/*jpg")
    
    fileNames = []    
    for name in names:
        fileNames.append(name)
    
    return fileNames

# リサイズしてnumpy配列で返す
def getTrainData(height, width):
    fileNames = getAllFileName()    
    
    imgs = np.empty((0, height, width, 3), int)
    
    for name in fileNames:
        img = Image.open(name)
        img = img.resize((height, width))
        img = np.asarray(img)
        img = img.reshape(-1, height, width, 3)
        imgs = np.append(imgs, img, axis=0)
        
    return imgs

if __name__=='__main__':
    fileNames = getAllFileName()    
    # print(fileNames)
    
    imgs = getTrainData(height=64, width=64)
    
    plt.imshow(imgs[0])
    plt.show()