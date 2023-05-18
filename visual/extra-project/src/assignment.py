import zipfile
import os
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D
import numpy as np
import random

def unzipping_files():
    print('unzipping')
    with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
        zip_ref.extractall() # unzipping the file

def data_gen():
    batch_size = 40
    img_height = 150
    img_width = 150
    train_datagen = ImageDataGenerator(rescale = 1/255, validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale = 1/255)
    val_datagen = ImageDataGenerator(rescale = 1/255, validation_split=0.2)
    trainset = train_datagen.flow_from_directory(r"is that santa/train",target_size = (img_height,img_width),class_mode='binary',batch_size = batch_size, subset='training')
    testset = test_datagen.flow_from_directory(r"is that santa/test",target_size = (img_height,img_width),class_mode = 'binary',batch_size = batch_size)
    valset = val_datagen.flow_from_directory(r"is that santa/train",target_size = (img_height,img_width),class_mode = 'binary',batch_size = batch_size, subset='validation')
    return trainset, testset, valset, batch_size, img_height, img_width

def print_pic(batch_size, img_height, img_width):
    plt.figure(figsize=(10, 10))
    class_names = ["not-a-santa", "santa"]
    train_pic = tensorflow.keras.utils.image_dataset_from_directory(
        "is that santa/train",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    for images, labels in train_pic.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.savefig('out/examples.png')

def training(trainset, testset, valset, batch_size):
    model = Sequential([Conv2D(16,(3,3),input_shape = (150,150,3),activation = 'relu'),
                                 MaxPooling2D(2,2),
                                 Conv2D(32,(3,3),activation = 'relu'),
                                 MaxPooling2D(2,2),
                                 Conv2D(64,(3,3),activation = 'relu'),
                                 MaxPooling2D(2,2),
                                 Flatten(),
                                 Dense(512,activation = 'relu'),
                                 Dense(1,activation = 'sigmoid')
    
        ])
    model.compile(optimizer = "adam",loss = 'binary_crossentropy',metrics = ['accuracy'])
    H = model.fit(trainset,
            batch_size=batch_size,
            validation_data=valset,
            steps_per_epoch=len(trainset),
            validation_steps=len(valset),
            epochs=10,
            verbose=1)
    return model, H

def evaluate_model(H):
    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']

    loss = H.history['loss']
    val_loss = H.history['val_loss']

    epochs_range = range(10)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('out/eval.png')

def testing(model):
    input_options = ["N", "S"]
    for i in range(3):
        while True:
            try:
                user_input = input("Choose a category (non-santa: N / santa: S): ") # asking the user for input
                if user_input in input_options:
                    if user_input == "S":
                        input_category = "santa"
                    else:
                        input_category = "not-a-santa"
                    break
                else:
                    print("Please select either S or N") # if the answer isnt valid, ask again
            except ValueError:
                print('Thats not a valid answer. Please try Again.')
                continue
        dir = os.path.join("is that santa/test/", input_category)
        random_pic_name = random.choice(os.listdir(dir)) # choose image randomly
        random_pic_path = os.path.join(dir, random_pic_name)
        random_pic = image.load_img(random_pic_path,target_size = (150,150)) # load image
        random_pic = image.img_to_array(random_pic)
        random_pic = np.expand_dims(random_pic, axis=0)
        images = np.vstack([random_pic])
        answer = model.predict(images) # predict
        print('Prediction score: ', answer[0][0])
        if answer[0][0] == 0:
            print("Filepath: ", random_pic_path)
            print("It is not a Santa:(")
        elif answer[0][0] == 1:
            print("Filepath: ", random_pic_path)
            print("It is a Santa!!:)")
        else:
            print("The training didn't work. Please run the code again.")

def main():
    if not os.path.exists("is that santa"):
        unzipping_files()
    trainset, testset, valset, batch_size, img_height, img_width = data_gen()
    print_pic(batch_size, img_height, img_width)
    model, H = training(trainset, testset, valset, batch_size)
    evaluate_model(H)
    testing(model)

if __name__ == '__main__':
    main()