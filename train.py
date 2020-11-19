import os
import cv2
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from models.vgg16 import VGG16
from models.lenet import LeNet
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")

dataset='dataset'
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="type the path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="type the path to output loss/accuracy plot")
args = vars(ap.parse_args())

epochs=25
learning_rate=1e-3
batch_size=64

data=[]
labels=[]

imagePaths=sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image=cv2.imread(imagePath)
    image=cv2.resize(image,(224,224))
    image=img_to_array(image)
    data.append(image)
    label=imagePath.split(os.path.sep)[-2]
    label=1 if label =='covid' else 0
    labels.append(label)

data=np.array(data,dtype="float")/255.0
labels=np.array(labels)

(trainX,testX,trainY,testY)= train_test_split(data, labels, test_size = 0.25, random_state=42)

trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

#After preprocessing the data, now we bring in the model to train it with the data
print("[UPDATE]: We are now compiling the model.")
model = LeNet.build(width=224, height=224, depth=3, classes=2)
opt = Adam(lr=learning_rate, decay=learning_rate / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
#Now that compiling the model is done, we will now train the data
print("[UPDATE]: We are now training the model.")
H = model.fit(x=aug.flow(trainX, trainY, batch_size=batch_size),
              validation_data=(testX, testY), steps_per_epoch=len(
                  trainX) // batch_size,
              epochs=epochs, verbose=1)

print("[UPDATE]: Training is done. We are serialising the network.")
model.save(args["model"], save_format="h5")

# We are now trying to visualise the performance of the model using matplotlib
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Slides/Not Slides")
plt.xlabel("Epoch ")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])