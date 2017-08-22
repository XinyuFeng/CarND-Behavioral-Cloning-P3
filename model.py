import csv
import cv2
import numpy as np
import sklearn
from sklean.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Lambda, Dropout, Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import h5py

def getPath(source_path):
    '''
    get the absolute image path
    '''
    filename = source_path.split('/')[-1]
    return '../data/IMG/' + filename

def load_img(path):
    '''
    load image from absolute path, then make an initial process (cropping and resize)
    :param path: image file's absolute path
    :return img: preprocessed image
    '''
    abs_path = getPath(path)
    img = cv2.imread(abs_path)
    img = img_preprocess(img)
    return img

def img_preprocess(img):
    '''
    transfrom BGR to RGB
    :param img:
    :return: n_img
    '''

    #n_img = img[50:140,:,:]
    #n_img = cv2.GaussianBlur(n_img, (3,3), 0)
    #n_img = cv2.resize(n_img, (200, 66), interpolation=cv2.INTER_AREA)
    n_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return n_img

def batch_generator(samples, batch_size):
    '''
    generate dataset with batch size by using generator
    '''
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
    '''
    randomly choose center, left or right image
    '''
    turn = np.random.rand()
    if turn < 0.33:
        return load_img(left), label + 0.2
    elif turn >= 0.33 and turn < 0.66:
        return load_img(right), label - 0.2
    return load_img(center), label

def rand_flip(image, steering):
    '''
    randomly flip an image
    '''
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering = - steering
    return image, steering

def augment(center, left, right, label): 
    '''
    augment methods used on image
    '''
    image, steering = rand_choose(center, left, right, label)
    image, steering = rand_flip(image, steering)
    return image, steering

def load_data():
    '''
    get train and validation dataset from csv
    '''
    samples = []
    with open('../data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[0] == "center":
                continue
            samples.append(line)
    shuffle(samples)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

def model_build():
    '''
    CNN model based on Nvidia car driving model
    '''
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dropout(0.5)) #add dropout to avoid overfitting
    model.add(Dense(100))
    model.add(Dropout(0.5)) #add dropout to avoid overfitting
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    return model

def model_train(model, train_samples, validation_samples):
    '''
    train model
    '''
    batch_size = 64
    #model.load_weights('model.h5')
    model.compile(loss='mse', optimizer=Adam(lr=1e-4))
    history_object = model.fit_generator(batch_generator(train_samples, batch_size),
                        #steps_per_epoch=len(train_samples) // batch_size,
                        validation_data=batch_generator(validation_samples, batch_size),
                        #validation_steps=len(validation_samples) // batch_size,
                        samples_per_epoch=len(train_samples),
                        nb_epoch=10,
                        nb_val_samples=len(validation_samples),
                        verbose=1)

    model.save('model1.h5')
    return history_object

def visualize_loss(history_object):
    '''
    visualize train loss and validation loss
    '''

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

def main():
    train_samples, validation_samples = load_data()
    model = model_build()
    history_object = model_train(model, train_samples, validation_samples)
    visualize_loss(history_object)



if __name__ == '__main__':
    main()