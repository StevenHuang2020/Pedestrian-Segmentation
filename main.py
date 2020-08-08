#python3 tensorflow 2.0  08/08/2020 
#Segmentation U-Net model using PennFudan pedestrain dataset
import tensorflow.keras as ks
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization,Conv2DTranspose,Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import argparse
import datetime

from loadImages import loadDataset

def prepareData():
    img_rows, img_cols = 256,256
    
    X, Y = loadDataset()
    print('X.shape=',X.shape)
    print('Y.shape=',Y.shape)
    
    X = X.reshape(X.shape[0], img_rows, img_cols, 3)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        
    print('X_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('X_test.shape = ', x_test.shape)
    print('Y_test.shape = ', y_test.shape)
    
    return x_train, y_train, x_test, y_test, (img_rows, img_cols, 1)

def unet(sz = (256, 256, 3)):
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
    
    
    lr = 1e-4
    opt = optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False)
    #opt = optimizers.Adadelta(learning_rate=lr, rho=0.95)
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9)
    #opt = optimizers.Adagrad(learning_rate=lr)    
    #opt = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #opt = optimizers.Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    #opt = optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    loss = ks.losses.binary_crossentropy
    #loss = ks.losses.categorical_crossentropy #one hot 
    model.compile(loss=loss,
                optimizer=opt,
                metrics=['accuracy'])
    
    model.summary()
    return model

def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', help = 'epochs')
    parser.add_argument('-new', dest='newModel', action='store_true')

    return parser.parse_args()

def main():
    arg = argCmdParse()
    
    epoch = 300
    newModel = False
    if arg.epoch:
        epoch = int(arg.epoch)
    if arg.newModel:
         newModel = True
         
    print('newModel=',newModel,'epoch=',epoch)

    x_train, y_train, x_test, y_test, input_shape = prepareData()
    print('input_shape = ', input_shape)

    modelName = r'./weights/trainPedSegmentation.h5'
    if newModel:
        model = unet()
    else: #continue trainning
        model = ks.models.load_model(modelName)
    
    log_dir = r"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = ks.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_filepath = r'./checkpoint'
    checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=False,save_freq=100)
        
    #model.fit(x_train, y_train, epochs=10, callbacks = [tensorboard_callback,checkpointer])
    model.fit(x_train, y_train, epochs=epoch, verbose=1, batch_size=100,
              validation_data=(x_test, y_test),callbacks = [tensorboard_callback,checkpointer]) #
    
    #score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1]
    
    model.save(modelName)
    model.save(r'./weights/' + 'trainPedSegmentation' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
    #model.save_weights(weightsFiles)
 
if __name__=='__main__':
    main()