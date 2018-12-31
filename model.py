import csv
import cv2    
import numpy as np

lines = []
data_path = './data/'
# These are some other data paths that were used during testing
#data_path = '../Origdata/'
#data_path = '../THdata/'
#data_path = '../THdata1/'

# Read the driving log CSV file
first_line = True
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if (first_line):
            # first line of the sample data has a header, just skip it
            first_line = False
        else:
            lines.append(line)

# Print some debug information to make sure we're on the right "track"
print(data_path + 'driving_log.csv')
# image path from the last line
print(line[0])
# steering value from the last line
print(line[3])

# define the arrays for images and (steering) measurements
images = []
measurements = []
# parse the CSV file data:
#   1) Read and save the images (3 perspectives - center, left, right)
#   2) Read the center steering measurement and produce a left and right steering measurement using a correction factor (tuning parameter)
#
first = True
for line in lines:
    # collect all three cameras (center, left, right)
    for i in range(3):
        source_path = line[i]
        tokens = source_path.split('/')
        #print('Tokens: ', tokens)
        filename = tokens[-1]
        local_path = data_path + 'IMG/' + filename
        #print(local_path)
        image = cv2.imread(local_path)
        images.append(image)
        # save the first images for report
        if (first):
            if (i == 0):
                cv2.imwrite('./First_Center.jpg', image)
            elif (i == 1):
                cv2.imwrite('./First_Left.jpg', image)
            else:
                cv2.imwrite('./First_Right.jpg', image)
    
    first = False
    
    left_correction = 0.15
    right_correction = 0.1
    measurement = float(line[3])
    # save center (steering) measurement
    measurements.append(measurement)
    # emulate left (steering) measurement
    measurements.append(measurement+left_correction)
    # emulate right (steering) measurement
    measurements.append(measurement-right_correction)
   
# Test - 2190 of each
#print(len(images))
#print(len(measurements))

# augment the data set by flipping the image and measurement along the vertical axis
augmented_images = []
augmented_measurements = []

first = True
for image, measurement in zip(images, measurements):
    # add the original image
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    # add the flipped image
    flipped_image = cv2.flip(image,1)
    flipped_measurement = float(measurement) * -1.0
    # save the first flipped image for report
    if (first):
        cv2.imwrite('./First_Flipped.jpg', flipped_image)

    first = False
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

# create the numpy arrays (required by Keras) from the augmented dataset
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# A little data check point
print(X_train.shape)

# My model starts here
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# This is an implementation of the NVidia model
model = Sequential()
# Normalization
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
# crop the top 70 pixels (sky) and bottom 25 pixels (car hood) - less data to process
model.add(Cropping2D(cropping=((70,25),(0,0))))
# 5 convolutional layers
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
# five fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# optimize with an Adam optimizer minimizing mean square error (mse)
model.compile(optimizer='adam', loss='mse')
# train the model using 20% of dataset for validation over 7 epochs shuffling the data each time
model.fit(X_train, y_train, validation_split=0.2, epochs=7, shuffle=True)

# save this model for use in the simulator
model.save('model.h5')
