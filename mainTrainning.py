#python3 tensorflow 2.0  08/08/2020 
#Segmentation U-Net model using PennFudan pedestrain dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #close tf debug info

import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization,Conv2DTranspose,Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

import numpy as np
import argparse
import datetime

from makeImagesDB import loadDataset
print('Maintraining start...')

def prepareData():
    img_rows, img_cols = 256,256
    
    X, Y = loadDataset()
    print('X.shape=',X.shape)
    print('Y.shape=',Y.shape)
    
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=12) #insure every time has same train and test dataset.
        
    print('X_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('X_test.shape = ', x_test.shape)
    print('Y_test.shape = ', y_test.shape)
    
    return x_train, y_train, x_test, y_test, (img_rows, img_cols, 1)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return 1.0 - dice_coef(y_true, y_pred)

def unet(sz = (256, 256, 1)):
    x = Input(sz)
    inputs = x
    
    #down sampling 
    f = 8
    layers = []
    
    for i in range(0, 6):
        x = Conv2D(f, 3, activation='relu', padding='same') (x)
        x = BatchNormalization()(x)
        x = Conv2D(f, 3, activation='relu', padding='same') (x)
        x = BatchNormalization()(x)
        layers.append(x)
        x = MaxPooling2D() (x)
        x = BatchNormalization()(x)
        f = f*2
    ff2 = 64
    
    #bottleneck 
    j = len(layers) - 1
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = BatchNormalization()(x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    x = BatchNormalization()(x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j -1 
    
    #upsampling 
    for i in range(0, 5):
        ff2 = ff2//2
        f = f // 2 
        x = Conv2D(f, 3, activation='relu', padding='same') (x)
        x = BatchNormalization()(x)
        x = Conv2D(f, 3, activation='relu', padding='same') (x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
        x = BatchNormalization()(x)
        x = Concatenate(axis=3)([x, layers[j]])
        j = j -1 
        
    #classification 
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = BatchNormalization()(x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = BatchNormalization()(x)
    outputs = Conv2D(1, 1, activation='sigmoid') (x)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    
    lr = 1e-5
    #opt = optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False)
    #opt = optimizers.Adadelta(learning_rate=lr, rho=0.95)
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9)
    #opt = optimizers.Adagrad(learning_rate=lr)    
    opt = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #opt = optimizers.Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    #opt = optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    #loss = ks.losses.binary_crossentropy
    #loss = ks.losses.categorical_crossentropy #one hot 
    #loss = ks.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(#loss=loss,
                  loss = dice_coef_loss,
                optimizer=opt,
                metrics=['accuracy'])
    
    model.summary()
    plotModel(model)
    return model

def plotModel(model,dstFile=r'model.png',show_shapes=True):
    plot_model(model, to_file=dstFile, show_shapes=show_shapes)
    
def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', help = 'epochs')
    parser.add_argument('-new', dest='newModel', action='store_true')
    parser.add_argument('-up', dest='setLearningRate', action='store_true')
    return parser.parse_args()

def updateLearningRate(model, lr=1e-4):
    print('Orignial lr:',K.get_value(model.optimizer.lr))
    # To set learning rate
    K.set_value(model.optimizer.lr, lr)
    
def loadModel(modelName):
    return ks.models.load_model(modelName,custom_objects={'dice_coef_loss': dice_coef_loss})

def main():
    arg = argCmdParse()
    
    epoch = 20
    newModel = False
    setLr = False
    if arg.epoch:
        epoch = int(arg.epoch)
    if arg.newModel:
         newModel = True
    if arg.setLearningRate:
        setLr = True
        
    print('newModel=',newModel,'epoch=',epoch,'setLr=',setLr)

    x_train, y_train, x_test, y_test, input_shape = prepareData()
    print('input_shape = ', input_shape)

    modelName = r'./weights/trainPedSegmentation.h5'
    if newModel:
        model = unet()
    else: #continue trainning
        model = loadModel(modelName) #ks.models.load_model(modelName,custom_objects={'dice_coef_loss': dice_coef_loss})
        if setLr:
            updateLearningRate(model,lr=1e-4)
        
    log_dir = r"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = ks.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_filepath = r'./checkpoint'
    checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=False,save_freq=1000)
        
    #model.fit(x_train, y_train, epochs=10, callbacks = [tensorboard_callback,checkpointer])
    model.fit(x_train, y_train, epochs=epoch, 
              verbose=1, 
              batch_size=160,
              validation_data=(x_test, y_test),
              #callbacks = [tensorboard_callback,checkpointer]
              callbacks = [tensorboard_callback]
              ) #
    
    #score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1]
    
    model.save(modelName)
    model.save(r'./weights/' + 'trainPedSegmentation' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
    #model.save_weights(weightsFiles)
 
if __name__=='__main__':
    main()