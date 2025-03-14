import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
import random,shutil
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# A generator is created for image augmentation. This helps in increasing the data and also helps making the model more robust
def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS= 32 #Batch_Size
TS=(24,24) #Target_size
train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('data/valid',shuffle=True, batch_size=BS,target_size=TS)
SPE= len(train_batch.classes)//BS #Steps_per_Epochs
VS = len(valid_batch.classes)//BS #Validation_size
print(SPE,VS)


# img,labels= next(train_batch)
# print(img.shape)


# 32 convolution filters used, each of size 3x3
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),

    # Another convolutional layer with 32 filters
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

    # 64 convolution filters used, each of size 3x3
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

    # Dropout layer to prevent overfitting
    Dropout(0.25),

    # Flattening the input for the fully connected layers
    Flatten(),
    
    # Fully connected layer with 128 neurons
    Dense(128, activation='relu'),
    Dropout(0.5),

    # Output layer with 2 neurons (Open/Closed classification)
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=SPE, validation_steps=VS)

model.save('models/cnnCat2.h5', overwrite=True)