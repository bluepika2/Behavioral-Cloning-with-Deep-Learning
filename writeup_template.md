# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/raw.png "raw"
[image2]: ./output_images/rgb.png "rgb"
[image3]: ./output_images/cropped.png "cropped"
[image4]: ./output_images/resized.png "resized"
[image5]: ./output_images/flipped.png "flipped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 32 and 64 as below code. The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 
```python
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
```
#### 2. Attempts to reduce overfitting in the model

To prevent overfitting, I leveraged a couple of data augmentation skills such as flipping images from center camera, as well as using left and right image to make a model more generalized. The model was trained and validated on different data set, and shuffled to ensure that the model is not overfitting. The model was tested in simulator in order to confirm the vehicle could stay within the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Batch size was intially set to 64, which can reduce validation loss because the model will learn more often. However, because of file size limit for submission, I used 128 batch size afterwards.

#### 4. Appropriate training data

Actually, I utilized the sample data because I was a really bad driver within Simulator environment... I could not drive the car within the track..

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
```

#### 3. Creation of the Training Set & Training Process

In order to create the training data, I used the Udacity sample data as a base. For each image, normalization was applied before the image was imported into the network. In my case, a training sample is composed of following four images:
1. Center camera image
2. Center camera image flipped horizontally
3. Left camera image
4. Right camera image

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