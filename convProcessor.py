import datasets
import numpy as np
import matplotlib.pyplot as plt
import models
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


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

trainX=[]
for i in range(len(trainY)):
    id=trainIDs[i]
    trainX.append(images[i])
trainX=np.array(trainX)

testX=[]
for i in range(len(testY)):
    id=testIDs[i]
    testX.append(images[i])
testX=np.array(testX)


model=models.create_conv_regress(trainX[0].shape)
opt = Adam(lr=1e-3, decay=1e-3 / 200)    # lr --> Learnrate   decay ---> absenken der Lernrate nach jeder Epoche
model.compile(loss="mean_absolute_percentage_error", optimizer='rmsprop')

history=model.fit(trainX, trainY, validation_data=(testX, testY),
          epochs=1000, batch_size=20)

score=model.evaluate(trainX, trainY, verbose=0)
print('Test loss:', score)

'''
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
'''

