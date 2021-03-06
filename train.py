from keras.layers import Input
from keras.models import Model,load_model,save_model
from keras.layers import Activation,BatchNormalization
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

import albumentations as A

#!pip3 install albumentations


#read template image
img = cv2.imread("template.jpg")
rows, cols, ch = img.shape


width=128
height=128

batch_size = 64

num_epochs = 100

continue_train = True


val_score1 = .5 #np.load('val_score1.npy')


dim = (width, height)
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 


def getdata(img,count,disp=False):
    #randomly transform the template image to generate sample data
    #need to add a new function that generates data by registering real -
    #input images with template and thus generating transform matrix
    xtrain=[]
    ytrain=[]
    d=20
    
    pts1 = np.float32([[0, 0], [width, 0], [width, height]])

    '''
    cv2.circle(img, tuple(pts1[0]), 5, (0, 0, 255), -1)
    cv2.circle(img, tuple(pts1[1]), 5, (0, 0, 255), -1)
    cv2.circle(img, tuple(pts1[2]), 5, (0, 0, 255), -1)
    cv2.imshow("Image", img)
    '''
    aug = A.Compose([
    A.RandomBrightnessContrast(p=.5),    
    A.RandomGamma(p=.5), 
    A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=.5),
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, 
                        shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1),p=.5),
                        
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, alpha_coef=0.1, p=.5),
    #A.RandomSunFlare(flare_roi=(0, 0, 0.2, 0.2), angle_lower=0.2, p=.5),
    #A.CLAHE(p=1), 
    A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=50, val_shift_limit=50, p=.5),
    
    ], p=.8)
    
    for i in range(count):
        

        d0 = random.sample(range(-d, d), 2)
        d1 = random.sample(range(-d, d), 2)
        d2 = random.sample(range(-d, d), 2)
        #import pdb;pdb.set_trace()

        pts2 = np.float32([pts1[0]+d0, pts1[1]+d1, pts1[2]+d2])

        matrix = cv2.getAffineTransform(pts1, pts2)
        result = cv2.warpAffine(img, matrix, (width, height))
        #augmentations
        result = aug(image=result)['image']
        
        #import pdb;pdb.set_trace()
        matrix = cv2.invertAffineTransform(matrix)
        matrix = matrix.flatten()
        
        xtrain.append(result)
        ytrain.append(matrix)
        
        if disp==True:
            cv2.imshow("Affine transformation", result)
            cv2.waitKey(30)
    cv2.destroyAllWindows()
    xtrain = np.array(xtrain, dtype=np.float32)
    ytrain = np.array(ytrain, dtype=np.float32)
    return(xtrain/255.0, ytrain)



xtrain, ytrain = getdata(img,batch_size*1, disp=True)


xval, yval = getdata(img,batch_size*3, disp=True)


#from utils1 import get_initial_weights
#from layers import BilinearInterpolation

def localizer(input_shape=(width, height, 3), num_classes=10):
    image = Input(shape=input_shape)
    
    locnet = Conv2D(20, (5, 5))(image)
    locnet = BatchNormalization()(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Activation('relu')(locnet)
    
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = BatchNormalization()(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Activation('relu')(locnet)
    
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = BatchNormalization()(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Activation('relu')(locnet)
    
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = BatchNormalization()(locnet)
    #locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Activation('relu')(locnet)
    
    locnet = Conv2D(40, (5, 5))(locnet)
    locnet = BatchNormalization()(locnet)
    locnet = Activation('relu')(locnet)
    
    locnet = Flatten()(locnet)
    locnet = Dense(50)(locnet)
    locnet = Activation('relu')(locnet)
    
    theta = Dense(6)(locnet)
    
    return Model(inputs=image, outputs=theta)


if continue_train==True:
    model = load_model('weight.h5')
else:
    model = localizer()
    model.compile(loss='mean_squared_error', optimizer='adam')
    val_score1 =10000000
model.summary()


def transform(img,matrix):
    #assuming input img is normalized, else remove 255s
    rows, cols, ch = img.shape
    #import pdb;pdb.set_trace()
    img = img*255
    img= img.astype(np.uint8)
    theta = matrix.reshape(2,3)
    img2 = cv2.warpAffine(img, theta, (cols, rows))
    img2= img2.astype(np.float32)
    img2 = img2/255.0
    return img2


def transform_plot(x_batch,y_batch, matrices):
    fig = plt.figure()
    plt.clf()
    for image_arg in range(3):
        #import pdb;pdb.set_trace()
        
        img = x_batch[image_arg]
        M2 = matrices[image_arg]
        M=y_batch[image_arg]
        
        img2 = transform(img,M2)
        
        print('    M:',[ '%.2f' % elem for elem in M ])
        print('   M2:',[ '%.2f' % elem for elem in M2 ])
        plt.subplot(3, 2, image_arg*2 + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(3, 2, image_arg*2 + 2)
        plt.imshow(img2)
        plt.axis('off')
    fig.canvas.draw()
    plt.show()



loss=[]
#val_score1=20

for epoch_arg in range(num_epochs):
    xtrain, ytrain = getdata(img,batch_size*16)
    for batch_arg in range(xtrain.shape[0]//batch_size):
        arg_0 = batch_arg * batch_size
        arg_1 = (batch_arg + 1) * batch_size
        x_batch, y_batch = xtrain[arg_0:arg_1], ytrain[arg_0:arg_1]
        
        #import pdb;pdb.set_trace()
        loss = model.train_on_batch(x_batch, y_batch)
    print('Epoch:',epoch_arg, '/',num_epochs, 'loss: ',loss)
    
    if (epoch_arg % 10 == 0) and (epoch_arg>0):
        val_score = model.evaluate(xval, yval, verbose=1)
        print('validation loss :', val_score)
        if val_score<val_score1:
            print('validation loss reduced, saving weights',val_score1, val_score)
            val_score1=val_score
            np.save('val_score1.npy',val_score1)
            model.save('weight.h5')
        
            matrices = model.predict_on_batch(x_batch)
            #transform_plot(x_batch,y_batch, matrices)
            print('-' * 40)
    print()


model = load_model('weight.h5')
matrices = model.predict_on_batch(xval)
transform_plot(xval,yval, matrices)


