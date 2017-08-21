import csv
import cv2
import numpy as np

def getPath(source_path):
    filename = source_path.split('/')[-1]
    return './data/IMG/' + filename

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[0] == "center":
            continue
        lines.append(line)

images = []
measurements = []
#print(lines)
for line in lines:
    measurement = float(line[3])
    # create adjusted steering measurements
    correction = 0.2
    left = measurement + correction
    right = measurement - correction

    img_center = cv2.imread(getPath(line[0]))
    img_left = cv2.imread(getPath(line[1]))
    img_right = cv2.imread(getPath(line[2]))

    images.append(img_center)
    images.append(img_left)
    images.append(img_right)

    measurements.append(measurement)
    measurements.append(left)
    measurements.append(right)
#print(images)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#split into train and valid dataset
portion = 0.8
partition = int(len(X_train) * portion)
X_train = X_train[0:partition]
y_train = y_train[0:partition]
X_valid = X_train[partition:len(X_train)]
y_valid = y_train[partition:len(y_train)]

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import h5py

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
batch_size = 100
history_object = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size),
                                     steps_per_epoch = X_train.shape[0] // batch_size,
                                     epochs = 10,
                                     validation_data=(X_valid, y_valid), verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')