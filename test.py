from keras.layers import Input
from keras.models import Model,load_model,save_model
from keras.layers import Activation,BatchNormalization
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense

import cv2
import numpy as np

def transform(img,matrix):
    #assuming input img is normalized, else remove 255s
    #import pdb;pdb.set_trace()
    rows, cols, ch = img.shape
    theta = matrix.reshape(2,3)
    img2 = cv2.warpAffine(img, theta, (cols, rows))
    return img2
    
if __name__ == '__main__':

    model = load_model('weight.h5')
    
    #read template image
    img = cv2.imread("test/IMG_20200120_210926.jpg")
    img0 = cv2.imread("template.jpg")
    rows, cols, ch = img.shape

    width=128
    height=128
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    img0 = cv2.resize(img0, dim, interpolation = cv2.INTER_AREA) 

    

    img1 = img/255.0
    img1 = np.expand_dims(img1, axis=0)

    matrix = model.predict(img1)

    img2 = transform(img,matrix)
    
    img_combined = cv2.addWeighted(img0, 0.5, img2, 0.5, 0)

    cv2.imshow('moved',img)
    cv2.imshow('matched',img2)
    cv2.imshow('template',img0)
    cv2.imshow('fused',img_combined)
    cv2.imwrite('results/matched.png',img2)
    cv2.imwrite('results/fused.png',img2)
    print('matrix',matrix)
    cv2.waitKey(0)



