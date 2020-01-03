import pickle
import numpy as np
import matplotlib.image as mpimg
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.optimizers import Adam
from keras.models import load_model


## my class takes leans on the NVIDIA Model contained in their paper http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf 

class SuperAwesomeDrivingAI:
    
    def __init__(self, lr, dc, model=None):
        self.lr = lr
        self.dc = dc
        
        if model == None:
            self.build_model()
        else:
            self.model = load_model(model)
        self.init_generator()
        
    def build_model(self):
        
        self.model = Sequential()
        
        # Input Normalization
        self.model.add(Lambda(lambda x:x / 255. -.5, input_shape=(80, 320, 3)))
        
        # Layer 1 - Conv2D
        self.model.add(Conv2D(
            filters=16,
            kernel_size=(5,5),
            strides=(2,2),
            padding='same',
            activation='relu'
            ))
        self.model.add(AveragePooling2D(
            pool_size=(2,2)))
        self.model.add(Dropout(0.5)) 

        # Layer 2 - Conv2D
        self.model.add(Conv2D(
            filters=32,
            kernel_size=(5,5),
            strides=(2,2),
            padding='same',
            activation='relu'))
        self.model.add(AveragePooling2D(
            pool_size=(2,2)))
        self.model.add(Dropout(0.5)) 
        
        # Layer 3 - Conv2D
        self.model.add(Conv2D(
            filters=64,
            kernel_size=(5,5),
            strides=(2,2),
            padding='same',
            activation='relu'))
        self.model.add(Dropout(0.5))         
        
        # Layer 4 - Conv2D
        self.model.add(Conv2D(
            filters=128,
            kernel_size=(3,3),
            padding='same',
            activation='relu'))
        
        
        # Layer 5 - Dense
        self.model.add(Flatten())
        self.model.add(Dense(500))
        self.model.add(Dropout(0.2))
        
        # Layer 6 - Dense
        self.model.add(Dense(100))
        self.model.add(Dropout(0.2))
        
        # Layer 7 - Dense
        self.model.add(Dense(20))
        self.model.add(Dropout(0.2))
        
        # Layer 8 - Dense
        self.model.add(Dense(1))
        
        # Model compile
        self.opt = Adam(lr=self.lr, decay=self.dc)

        self.model.compile(loss='mse',
                      optimizer=self.opt,
                      metrics=['mse'])

    def save_model(self):
        self.model.save('model.h5')
    
    def init_generator(self):

        self.IDG = ImageDataGenerator(
                        rotation_range=10,
                        width_shift_range=.1,
                        shear_range=.1,
                        zoom_range=.1,
                        vertical_flip=False,
                        validation_split=.9
        )
        
    def feed_batches(self,x,y,EPOCHS = 5, BATCH_SIZE = 1028):
        
        for EPOCH in range(EPOCHS):
            print('Epoch', EPOCH)
            batches = 0
            for x_batch, y_batch in self.IDG.flow(x, y, batch_size=BATCH_SIZE):
                self.model.fit(x_batch, y_batch, verbose=0)
                batches += 1
                if batches >= len(x) / BATCH_SIZE:
                    break
        


model = SuperAwesomeDrivingAI(1e-4, 1e-6, 'model.h5')

# training on smooth driving around the track

#df = pd.read_csv('imgs/log.csv', header=0)
#imgs = np.array([mpimg.imread(img) for img in df['path']])
#label = np.array([label for label in df['label']])
#model.feed_batches(imgs, label, 10)

# additional training for more aggressive curves, forgot to crop the images

df = pd.read_csv('add_imgs/log.csv', header=0)
imgs = np.array([mpimg.imread(img)[50:130,:,:] for img in df['path']])
label = np.array([label for label in df['label']])
model.feed_batches(imgs, label, 5)

model.save_model()