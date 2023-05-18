import sys
sys.path.append("..")
import cv2
from zipfile import ZipFile
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# predefining function

def plot_histogram(filename):
    input_path = os.path.join("flowers", filename) # set data path
    image = cv2.imread(input_path) #load image
    channels = cv2.split(image) #split channels
    colors = ("b", "g", "r") # names of colours
    plt.figure() # create plot
    plt.title("Histogram") # add title
    plt.xlabel("Bins") # add xlabel
    plt.ylabel("# of Pixels") # add ylabel
    for (channel, color) in zip(channels, colors): # for every tuple of channel, colour
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256]) # create a histogram
        plt.plot(hist, color=color) # plot histogram
        plt.xlim([0, 256]) # set limits of x-axis
    plt.show() # show plot

# running my functions

def unzipping():
    file_name = "data/flowers.zip"
    with ZipFile(file_name, 'r') as zip:
        zip.printdir() # printing all the contents of the zip file
        print('Extracting all the files now...')
        zip.extractall() # extracting all the files
        print('Done!')

def one_colour_hist():
    path_to_image = os.path.join("flowers", "image_0013.jpg") # randomly choosing an image
    image = cv2.imread(path_to_image) # defining that one randomly chosen image
    # blue
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    plt.plot(hist, color="Blue")
    # green
    hist = cv2.calcHist([image], [1], None, [256], [0,256])
    plt.plot(hist, color="Green")
    # red
    hist = cv2.calcHist([image], [2], None, [256], [0,256])
    plt.plot(hist, color="Red")
    # compare
    hist_mainpic = cv2.calcHist([image], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
    hist_mainpic = cv2.normalize(hist_mainpic, hist_mainpic, 0, 1.0, cv2.NORM_MINMAX)
    return hist_mainpic

def all_colour_hist():
    matplotlib.use('Agg') # hiding the plots so it wont be too long
    directory = "flowers" # defining the directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f): # checking if it is a file
            plot_histogram(filename) # plotting
    return directory

def compare_hist(directory, hist_mainpic):
    hist_all = pd.DataFrame(columns = ['Filename', 'Histogram_diff']) # create an empty df
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f): # checking if it is a file
            other_image = cv2.imread(f)
            plot_histogram(filename) # plot
            hist_otherpic = cv2.calcHist([other_image], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
            hist_otherpic = cv2.normalize(hist_otherpic, hist_otherpic, 0, 1.0, cv2.NORM_MINMAX)
            diff = round(cv2.compareHist(hist_mainpic, hist_otherpic, cv2.HISTCMP_CHISQR), 2)
            new_data = (filename, diff)
            new_df_line = pd.DataFrame([new_data], columns=['Filename', 'Histogram_diff'])
            hist_all = pd.concat([hist_all, new_df_line], ignore_index=True)
    return hist_all

def five_images(hist_all):
    top_five = hist_all.nlargest(5, 'Histogram_diff')
    df_top_five = pd.DataFrame(columns = ['Filename', 'Distance']) # create an empty df
    df_top_five['Filename'] = top_five['Filename'] # assigning data to column
    df_top_five['Distance'] = top_five['Histogram_diff'] # assigning data to column
    df_top_five.to_csv('out/top_five.csv') # saving it as a csv

def main():
    if len(os.listdir("flowers")) == 0:
        unzipping()
    hist_mainpic = one_colour_hist()
    directory = all_colour_hist()
    hist_all = compare_hist(directory, hist_mainpic)
    five_images(hist_all)

if __name__ == '__main__':
    main()