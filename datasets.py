# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import glob
import cv2
import os
from PIL import Image
import numpy as np
from skimage.io import imread_collection
from skimage.transform import resize


def loadFrontalImages(number=-1, shape=(50, 50)):
    return loadImages('frontal', number, shape)

def loadImages(type, number=-1, shape=(50, 50)):
    # your path
    col_dir = 'data/*'+type+'.jpg'
    # creating a collection with the available images
    col = imread_collection(col_dir)
    img = []
    if number == -1:
        number = len(col)
    for i in range(number):
        img.append(resize(col[i], (shape[0], shape[1])))
    return np.array(img)

def load_house_attributes(inputPath="./data/HousesInfo.txt"):
    # initialize the list of column names in the CSV file and then
    # load it using Pandas
    df= pd.read_csv(inputPath, sep=" ")
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()

    # loop over each of the unique zip codes and their corresponding
    # count
    for (zipcode, count) in zip(zipcodes, counts):
        # the zip code counts for our housing dataset is *extremely*
        # unbalanced (some only having 1 or 2 houses per zip code)
        # so let's sanitize our data by removing any houses with less
        # than 25 houses per zip code
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)

    # return the data frame
    return df


def process_house_attributes(df, train, test):
    # initialize the column names of the continuous data
    continuous = ["bedrooms", "bathrooms", "area"]

    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])
    zipBinarizer = LabelBinarizer().fit(df["zipcode"])
    trainCategorical = zipBinarizer.transform(train["zipcode"])
    testCategorical = zipBinarizer.transform(test["zipcode"]) # Hot Vector der sechs verschiedenen ZIP-Codes in den Daten
    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    trainX = np.hstack([trainCategorical, trainContinuous]) # horizontales verknüpfen der beiden Arrays
    testX = np.hstack([testCategorical, testContinuous])
    # return the concatenated training and testing data
    return (trainX, testX)


def loadStackedImages(number=-1, shape=(50, 50)):
    bath=loadImages('bathroom', number, shape)
    frontal=loadImages('frontal', number, shape)
    bed=loadImages('bedroom', number, shape)
    kitchen=loadImages('kitchen', number, shape)
    result=np.concatenate((frontal, bath, bed, kitchen), axis=1)
    return result
