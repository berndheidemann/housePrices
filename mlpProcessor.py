from keras.optimizers import Adam
from keras.optimizers import Adadelta
from sklearn.model_selection import train_test_split
import datasets
import models
import numpy as np
import argparse
import locale
import os

print("[INFO] loading house attributes...")
data=datasets.load_house_attributes()
print("[INFO] constructing training/testing split...")
(train, test) = train_test_split(data, test_size=0.25, random_state=42)

maxPrice = train["price"].max()
trainY = train["price"] / maxPrice
testY = test["price"] / maxPrice

print("[INFO] processing data...")
(trainX, testX) = datasets.process_house_attributes(data, train, test)

model = models.create_mlp(trainX.shape[1], regress=True)
#opt = Adam(lr=1e-3, decay=1e-3 / 200)    # lr --> Learnrate   decay ---> absenken der Lernrate nach jeder Epoche
opt =Adadelta()

model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(trainX, trainY, validation_data=(testX, testY),
          epochs=300, batch_size=64)

score=model.evaluate(trainX, trainY, verbose=0)
print('Test loss:', score)

print("[INFO] predicting house prices...")
preds = model.predict(testX)

# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)
#print(absPercentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
print("[INFO] avg. house price: {}, std house price: {}".format(data["price"].mean(), data["price"].std()))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))