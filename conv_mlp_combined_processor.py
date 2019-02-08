import datasets
import numpy as np
import matplotlib.pyplot as plt
import models
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
#from livelossplot.keras import PlotLossesCallback
from keras import metrics
from PlotLoss import PlotLosses

def printImages(x,  columns = 5, rows = 5):
    ax = []
    w = 50
    h = 50
    fig = plt.figure(figsize=(9, 13))

    for i in range(columns*rows):
        image = x[i] * 255
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        plt.imshow(image.astype('uint8'))

    plt.show()

images=datasets.loadStackedImages(shape=(70, 100))
#printImages(images, 3, 3)
houseData=datasets.load_house_attributes()
(train, test) = train_test_split(houseData, test_size=0.25, random_state=42)
maxPrice = train["price"].max()
trainY = train["price"] / maxPrice
testY = test["price"] / maxPrice
trainY=np.array(trainY)
testY=np.array(testY)

trainIDs=train.index.values
testIDs=test.index.values

trainXConv=[]
for i in range(len(trainY)):
    id=trainIDs[i]
    trainXConv.append(images[i])
trainXConv=np.array(trainXConv)

testXConv=[]
for i in range(len(testY)):
    id=testIDs[i]
    testXConv.append(images[i])
testXConv=np.array(testXConv)

(trainXMlp, testXMlp) = datasets.process_house_attributes(houseData, train, test)


model=models.createCombined_cnn_mlp(trainXConv[0].shape, trainXMlp[0].shape)
opt = Adam(lr=1e-3, decay=1e-3 / 200)    # lr --> Learnrate   decay ---> absenken der Lernrate nach jeder Epoche
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

model.fit([trainXConv, trainXMlp], trainY, validation_data=([testXConv, testXMlp], testY),
                  epochs=500, batch_size=20, callbacks=[PlotLosses()])

score=model.evaluate([trainXConv, trainXMlp], trainY, verbose=0)
print('Test loss:', score)

