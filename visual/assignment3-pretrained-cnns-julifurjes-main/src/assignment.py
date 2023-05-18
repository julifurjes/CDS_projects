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
from sklearn.metrics import classification_report
import numpy as np
import shutil
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

# using the actual functions

def unzipping_files():
    print('unzipping')
    with zipfile.ZipFile('../../431824/archive.zip', 'r') as zip_ref:
        zip_ref.extractall('data') # unzipping the file into our data folder

def data_load():
    random.seed(45)
    train_df = pd.read_json('data/train_data.json', lines=True)
    test_df = pd.read_json('data/test_data.json', lines=True)
    val_df = pd.read_json('data/val_data.json', lines=True)
    def convert_image_path(image_path):
        base_dir = "data"
        return os.path.join(base_dir, image_path)
    train_df = train_df.sample(n=1000) # choosing only 1.000 datapoints
    test_df = test_df.sample(n=500) # choosing only 500 datapoints
    val_df = val_df.sample(n=200) # choosing only 200 datapoints
    train_df['image_path'] = train_df['image_path'].apply(convert_image_path)
    test_df['image_path'] = test_df['image_path'].apply(convert_image_path)
    val_df['image_path'] = val_df['image_path'].apply(convert_image_path)
    labelNames = np.unique(np.array(train_df['class_label'])) # creating a list of all the labels
    return train_df, test_df, val_df, labelNames

def image_gen(train_df, test_df, val_df):
    img_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
    batch_size = 32
    target_size = (224,224)
    train_images = img_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='class_label',
        target_size=target_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        seed=42,
    )
    val_images = img_generator.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='class_label',
        target_size=target_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        seed=42,
    )
    test_images = img_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col='class_label',
        target_size=target_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False,
        seed=42,
    )
    return img_generator, train_images, val_images, test_images

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
    output = Dense(15, activation='softmax')(class1) # c = (new)(b)
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

def training(model, train_images, val_images):
    batch_size = 64
    H = model.fit(train_images,
            batch_size=batch_size,
            validation_data=val_images,
            steps_per_epoch=train_images.samples // batch_size,
            validation_steps=val_images.samples // batch_size,
            epochs=10,
            verbose=1)
    return H

def evaluating(H, model, test_images, labelNames):
    plot_history(H, 10)
    predictions = model.predict(test_images, batch_size=500)
    report = classification_report(test_images.classes, 
                            predictions.argmax(axis=1),
                            target_names=labelNames)
    f = open("out/classification_report.txt", "w") 
    f.write(report) # saving the classifier metrics as a txt file
    f.close() # closing the txt file

def main():
    if len(os.listdir("data")) == 0:
        unzipping_files()
    train_df, test_df, val_df, labelNames = data_load()
    img_generator, train_images, val_images, test_images = image_gen(train_df, test_df, val_df)
    model = model_creation()
    H = training(model, train_images, val_images)
    evaluating(H, model, test_images, labelNames)

if __name__ == '__main__':
    main()