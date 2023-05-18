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
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random

# predefining plotting function

def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig('out/history.png')

def unzipping_files():
    print('unzipping')
    with zipfile.ZipFile('../../431824/archive.zip', 'r') as zip_ref:
        zip_ref.extractall('data') # unzipping the file into our data folder

def data_load():
    random.seed(45)
    train_df = pd.read_json('data/train_data.json', lines=True)
    test_df = pd.read_json('data/test_data.json', lines=True)
    def convert_image_path(image_path):
        base_dir = "data"
        return os.path.join(base_dir, image_path)
    train_df = train_df.sample(n=500) # choosing only 5.000 datapoints
    test_df = test_df.sample(n=200) # choosing only 2.000 datapoints
    train_df['image_path'] = train_df['image_path'].apply(convert_image_path)
    test_df['image_path'] = test_df['image_path'].apply(convert_image_path)
    skencoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
    labels_to_vectorise = test_df.class_label.values
    y_test = skencoder.fit_transform(labels_to_vectorise.reshape(-1,1)) # converting the labels into vectors
    labels_to_vectorise = train_df.class_label.values
    y_train = skencoder.fit_transform(labels_to_vectorise.reshape(-1,1)) # converting the labels into vectors
    X_train = train_df['image_path'].to_numpy()
    X_test = test_df['image_path'].to_numpy()
    labelNames = np.unique(np.array(train_df['class_label'])) # creating a list of all the labels
    return train_df, test_df, X_train, X_test, y_train, y_test, labelNames

def prep_pic(X_train, X_test):
    new_X_train = [0] * len(X_train)
    for i in range(len(X_train)):
        new_X_train[i] = load_img(X_train[i], target_size=(224, 224))
        new_X_train[i] = img_to_array(new_X_train[i])
        new_X_train[i] = preprocess_input(new_X_train[i])
        new_X_train[i] = new_X_train[i].reshape((new_X_train[i].shape[0], new_X_train[i].shape[1], new_X_train[i].shape[2]))
    new_X_test = [0] * len(X_test)
    for i in range(len(X_test)):
        new_X_test[i] = load_img(X_test[i], target_size=(224, 224))
        new_X_test[i] = img_to_array(new_X_test[i])
        new_X_test[i] = preprocess_input(new_X_test[i])
        new_X_test[i] = new_X_test[i].reshape((new_X_test[i].shape[0], new_X_test[i].shape[1], new_X_test[i].shape[2]))
    return new_X_train, new_X_test

def labels(train_df, test_df, labelNames):
    print(labelNames)
    for label in labelNames:
        os.makedirs(os.path.join('data','images', 'train_labelled', label)) # creating empty folders for each label
    print(labelNames)
    for label in labelNames:
        os.makedirs(os.path.join('data','images', 'test_labelled', label)) # creating empty folders for each label
    
def copy_pic(train_df, test_df):
    for i in range(len(train_df)):
        dst_dir = os.path.join('data','images', 'train_labelled', train_df['class_label'].iloc[i])
        shutil.copy(train_df['image_path'].iloc[i], dst_dir) # distributing the pictures based on labels into folders
    for i in range(len(test_df)):
        dst_dir = os.path.join('data','images', 'test_labelled', test_df['class_label'].iloc[i])
        shutil.copy(test_df['image_path'].iloc[i], dst_dir) # distributing the pictures based on labels into folders

def model_creation():
    model = VGG16(include_top=False, 
              pooling='avg',
              input_shape=(32, 32, 3)) # loading the model (without class. layers)
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False # mark loaded layers as not trainable
    tf.keras.backend.clear_session()
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output) # a = (new)(old)
    class1 = Dense(128, activation='relu')(flat1) # b = (new)(a)
    output = Dense(10, activation='softmax')(class1) # c = (new)(b)
    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)
    # summarize
    model.summary()
    #compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)
    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()
    return model

def image_gen():
    train_dataset = datagen.flow_from_directory(
        'data/images/train_labelled/',
        target_size=(255, 255),
        subset="training",
        batch_size=32,
        class_mode='categorical')
    test_dataset = datagen.flow_from_directory(
        'data/images/test_labelled/',
        target_size=(255, 255),
        subset="training",
        batch_size=32,
        class_mode='categorical')
    return datagen, train_dataset, test_dataset

def training(model, new_X_train, new_X_test, y_train, y_test):
    datagen = ImageDataGenerator(rescale=1./255,
                            horizontal_flip=True, 
                            rotation_range=20)
    datagen.fit(new_X_train)
    H = model.fit(datagen.flow(new_X_train, y_train, batch_size=500), 
              validation_data = datagen.flow(new_X_test, y_test, batch_size=500, subset = "validation"),
            epochs=1)
    return H

def evaluating(H, X_test, y_test, labelNames):
    plot_history(H, 10)
    predictions = model.predict(X_test, batch_size=500)
    report = classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labelNames)
    f = open("out/classification_report.txt", "w") 
    f.write(report) # saving the classifier metrics as a txt file
    f.close() # closing the txt file

def main():
    #unzipping_files() - we dont want to run this again once its unzipped
    train_df, test_df, X_train, X_test, y_train, y_test, labelNames = data_load()
    new_X_train, new_X_test = prep_pic(X_train, X_test)
    #labels(train_df, test_df, labelNames) #we dont want to run this again
    #copy_pic(train_df, test_df) #we dont want to run this again
    model = model_creation()
    #datagen, train_dataset, test_dataset = image_gen()
    H = training(model, new_X_train, new_X_test, y_train, y_test)
    evaluating(H, X_test, y_test, labelNames)

main()