import numpy as np
import math
import csv
import cv2
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
# import sample data
with open('./data/driving_log.csv') as csvfile: # for training I moved this flie and images to my workspace, but after that I removed those files due to size limit for submission
    reader = csv.reader(csvfile)
    next(reader)
    samples = []
    for line in reader:
        samples.append(line)
        
# data split to train and validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# data augmentation
def data_augmentation(batch_sample):
    steering_angle = np.float32(batch_sample[3])
    images =[]
    steering_angles = []
    for image_path_index in range(3):
        image_name = batch_sample[image_path_index].split('/')[-1]
        image = cv2.imread('./data/IMG/' + image_name)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # image convert to BGR to RGB
        cropped_image = rgb_image[60:130,:] # image cropping for only region of interest
        resized_cropped_image = cv2.resize(cropped_image, (160, 70))
        images.append(resized_cropped_image)
        # considering from which image is coming (left, center, or right), and add or subtract 0.2 to steering angle at center
        if image_path_index == 1:
            steering_angles.append(steering_angle+0.2)
        elif image_path_index == 2:
            steering_angles.append(steering_angle-0.2)
        else:
            steering_angles.append(steering_angle)
            
        if image_path_index == 0:
            flipped_center_image = cv2.flip(resized_cropped_image, 1) # also flip image for center camera for better training
            images.append(flipped_center_image)
            steering_angles.append(-steering_angle)
    return images, steering_angles
# data generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                augmented_images, augmented_angles = data_augmentation(batch_sample)
                images.extend(augmented_images)
                steering_angles.extend(augmented_angles)
               
            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)
           
batch_size = 64

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Model Structure
model = Sequential()
model.add(Lambda(lambda x: (x/127.5) - 1.0, input_shape=(70,160,3)))
model.add(Conv2D(24, (5, 5), activation='relu'))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.5))
model.add(Conv2D(36, (5, 5), activation='relu'))
model.add(Conv2D(48, (5, 5), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=3, verbose=1)
model.save('model.h5')