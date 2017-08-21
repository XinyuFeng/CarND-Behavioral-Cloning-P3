import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import h5py

def getPath(source_path):
    filename = source_path.split('/')[-1]
    return '../data/IMG/' + filename

def load_img(path):
    abs_path = getPath(path)
    return cv2.imread(abs_path)

def batch_generator(samples, batch_size):
    num_samples = len(samples)
    while True:

        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center, left, right = batch_sample[0], batch_sample[1], batch_sample[2]
                label = float(batch_sample[3])
                image, steering_angle = augment(center, left, right, label)
                images.append(image)
                angles.append(steering_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            #print(X_train)
            yield sklearn.utils.shuffle(X_train, y_train)

def rand_choose(center, left, right, label):
    turn = np.random.rand()
    if turn < 0.33:
        return load_img(left), label + 0.2
    elif turn >= 0.33 and turn < 0.66:
        return load_img(right), label - 0.2
    return load_img(center), label

def rand_flip(image, steering):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering = - steering
    return image, steering

def augment(center, left, right, label):
    image, steering = rand_choose(center, left, right, label)
    image, steering = rand_flip(image, steering)
    return image, steering

def load_data():
    samples = []
    with open('../data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[0] == "center":
                continue
            samples.append(line)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

def trash():
    lines = []
    #with open('./data/driving_log.csv') as csvfile:
     #   reader = csv.reader(csvfile)
      #  for line in reader:
       #     if line[0] == "center":
        #        continue
         #   lines.append(line)

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

def model_build():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Conv2D(64, 3, 3, activation="relu"))
    model.add(Conv2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    return model

def model_train(model, train_samples, validation_samples):
    batch_size = 128
    model.compile(loss='mse', optimizer=Adam(lr=1e-4))
    model.fit_generator(batch_generator(train_samples, batch_size),
                        steps_per_epoch=len(train_samples) // batch_size,
                        validation_data=batch_generator(validation_samples, batch_size),
                        validation_steps=len(validation_samples) // batch_size,
                        epochs=10,
                        verbose=1)

    model.save('model.h5')

def main():
    train_samples, validation_samples = load_data()
    model = model_build()
    model_train(model, train_samples, validation_samples)


if __name__ == '__main__':
    main()