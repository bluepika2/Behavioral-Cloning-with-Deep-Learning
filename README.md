# Behavioral Cloning Project

Overview
---

In this project, I will use what I've learned about deep neural networks and convolutional neural networks to clone driving behavior. I will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.
Data collection will be done within a simulator environment. I'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track in a simulator.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./output_images/raw.png "raw"
[image2]: ./output_images/rgb.png "rgb"
[image3]: ./output_images/cropped.png "cropped"
[image4]: ./output_images/resized.png "resized"
[image5]: ./output_images/flipped.png "flipped"

My project includes the following files:
* model.py containing the code to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 containing video of self-driving results based on trained neural network

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 32 and 64 as below code. The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 
```python
model = Sequential()
model.add(Lambda(lambda x: (x/127.5) - 1.0, input_shape=(70,160,3)))
model.add(Conv2D(24, (5, 5), activation='relu'))
model.add(Conv2D(36, (5, 5), activation='relu'))
model.add(Conv2D(48, (5, 5), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
```
#### 2. Attempts to reduce overfitting in the model

To prevent overfitting, I leveraged a couple of data augmentation skills such as flipping images from center camera, as well as using left and right image to make a model more generalized. The model was trained and validated on different data set, and shuffled to ensure that the model is not overfitting. The model was tested in simulator in order to confirm the vehicle could stay within the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Batch size was intially set to 64, which can reduce validation loss because the model will learn more often. However, because of file size limit for submission, I used 128 batch size afterwards.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a good model was to refer to the Nvidia's architecture since this model has been proved to be very successful in self-driving car tasks. Therefore, I utilized this architecture, and also intially tried Maxpooling, and Dropout method, but it turns out those extra methods could cause worse behavior for the car. Those two methods could not prevent overfitting because the model was still under fitting, so I did not use those two methods.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Since I was using data augmentation skills, the mean squared error was low at both on the training and validation steps.
One of the challenging parts for me was to make a change for `drive.py`. I already did data augmentation when I trained the model, so image input size was different from default image size from simulator. Eventually, I needed to add data augmentation in `drive.py` as well.
#### 2. Final Model Architecture

The final model architecture (model.py lines 73-86) consisted of a convolution neural network with fully connected layers as below.
```python
model = Sequential()
model.add(Lambda(lambda x: (x/127.5) - 1.0, input_shape=(70,160,3)))
model.add(Conv2D(24, (5, 5), activation='relu'))
model.add(Conv2D(36, (5, 5), activation='relu'))
model.add(Conv2D(48, (5, 5), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

In order to create the training data, I used the Udacity sample data as a base. For each image, normalization was applied before the image was imported into the network. In my case, a training sample is composed of following four images:
1. Center camera image
2. Center camera image flipped horizontally
3. Left camera image
4. Right camera image
Horizontally flipped camera image will help to prevent underfitting when the vehicle faces a different type of curve road. The pre-defined route in this simulator is just an oval track, so the camera is constantly seeing similar curve road such as only left curve or right curve. Without using flipped image, trained network will not know what to do when it encounters different curve road. Also, by using left and right camera image I can teach my model how to steer if the car drifts off to the left or the right.
Here are some examples of raw images:

![alt text][image1]

Here is converted image to RGB:

![alt text][image2]

Here is cropped image only for region of interest:

![alt text][image3]

Here is resized image:

![alt text][image4]

Here is flipped image from center camera:

![alt text][image5]

Here's a [link to my video result](./video.mp4)

After I preprocessed the image, I randomly shuffled that data set and put 20% of the data into a vaildation set.
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs for me was 5 as evidenced below mean square error. I used an adam optimizer so that manually training the learning rate wasn't necessary.
```
101/101 [==============================] - 205s - loss: 0.0300 - val_loss: 0.0179
Epoch 2/5
101/101 [==============================] - 189s - loss: 0.0161 - val_loss: 0.0204
Epoch 3/5
101/101 [==============================] - 190s - loss: 0.0149 - val_loss: 0.0178
Epoch 4/5
101/101 [==============================] - 189s - loss: 0.0136 - val_loss: 0.0147
Epoch 5/5
101/101 [==============================] - 190s - loss: 0.0125 - val_loss: 0.0156
```
