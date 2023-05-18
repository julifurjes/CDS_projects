import zipfile
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
import numpy as np
import shutil

def unzipping_files():
    print('unzipping')
    with zipfile.ZipFile('../../431824/archive.zip', 'r') as zip_ref:
        zip_ref.extractall('data') # unzipping the file into our data folder

def data_load():
    y_train = pd.read_json('data/train_data.json', lines=True)
    y_test = pd.read_json('data/test_data.json', lines=True)
    def convert_image_path(image_path):
        base_dir = "data"
        return os.path.join(base_dir, image_path)
    y_train = y_train.sample(n=10000) # choosing only 10.000 datapoints
    y_train['image_path'] = y_train['image_path'].apply(convert_image_path)
    print(y_train)
    y_test['image_path'] = y_test['image_path'].apply(convert_image_path)
    return y_train, y_test

def labels(y_train, y_test):
    labelNames = np.unique(np.array(y_train['class_label'])) # creating a list of all the labels
    print(labelNames)
    for label in labelNames:
        os.makedirs(os.path.join('data','images', 'train_labelled', label)) # creating empty folders for each label
    labelNames = np.unique(np.array(y_test['class_label'])) # creating a list of all the labels
    print(labelNames)
    for label in labelNames:
        os.makedirs(os.path.join('data','images', 'test_labelled', label)) # creating empty folders for each label
    
def copy_pic(y_train, y_test):
    for i in range(len(y_train)):
        dst_dir = os.path.join('data','images', 'train_labelled', y_train['class_label'].iloc[i])
        shutil.copy(y_train['image_path'].iloc[i], dst_dir) # distributing the pictures based on labels into folders
    for i in range(len(y_test)):
        dst_dir = os.path.join('data','images', 'test_labelled', y_test['class_label'].iloc[i])
        shutil.copy(y_test['image_path'].iloc[i], dst_dir) # distributing the pictures based on labels into folders

def image_gen():
    datagen = ImageDataGenerator(rescale=1./255,
                            horizontal_flip=True, 
                            rotation_range=20)
    train_dataset = datagen.flow_from_directory(
        'data/images/train_labelled/',
        target_size=(255, 255),
        subset="training",
        batch_size=32,
        class_mode='categorical')
    test_dataset = datagen.flow_from_directory(
        'data/images/test_labelled/',
        target_size=(255, 255),
        batch_size=32,
        class_mode='categorical')

def classification(X_train, y_train):
    model = VGG16(include_top=False, 
              pooling='avg',
              input_shape=(32, 32, 3)) # loading the model (without class. layers)
    for layer in model.layers:
        layer.trainable = False # mark loaded layers as not trainable
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output) # a = (new)(old)
    class1 = Dense(128, activation='relu')(flat1) # b = (new)(a)
    output = Dense(10, activation='softmax')(class1) # c = (new)(b)
    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)
    # summarize
    model.summary()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)
    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    H = model.fit(X_train, y_train, # train
            validation_split=0.1,
            batch_size=128,
            epochs=10,
            verbose=1)
    plot_history(H, 10)

def main():
    #unzipping_files() - we dont want to run this again once its unzipped
    y_train, y_test = data_load()
    #labels(y_train, y_test) - we dont want to run this again
    #copy_pic(y_train, y_test) - we dont want to run this again
    image_gen()
    #classification(X_train, y_train)

main()