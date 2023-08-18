#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:31:51 2023

@author: Sinyaev V.V.
"""

import os
import cv2
import json
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from DataGeneration1 import ImageGenerator

generator = ImageGenerator(output_folder='generated_images_NN')

data = []

shape_class_mapping = {
    'circle': 1,
    'triangle': 2,
    'rectangle': 3,
    'diamond': 4,
    'hexagon': 5
}

num_classes = len(shape_class_mapping)

image_folder = 'generated_images_2'
json_folder = 'generated_images_2'

for image_filename in os.listdir(image_folder):
    if image_filename.endswith('.png'):
        image_path = os.path.join(image_folder, image_filename)
        json_filename = image_filename.replace('.png', '.json')
        json_path = os.path.join(json_folder, json_filename)

        if os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                json_data = json.load(json_file)
            image = cv2.imread(image_path)
            data.append({'image': image, 'json_data': json_data})

input_shape = (256, 256, 3)

input_layer = Input(shape=input_shape, name='input')
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
class_output = Dense(num_classes, activation='softmax', name='class_output')(x)
coord_output = Dense(4, activation='linear', name='coord_output')(x)

model = Model(inputs=input_layer, outputs=[class_output, coord_output])

model.compile(optimizer='adam', loss={'class_output': 'categorical_crossentropy', 'coord_output': 'mean_squared_error'},
              loss_weights={'class_output': 1.0, 'coord_output': 1.0}, metrics=['accuracy'])

epochs = 1000
batch_size = 4

for epoch in range(epochs):
    print(f'epoch: {epoch}')
    epoch_losses = []
    epoch_accuracies = []
    for batch_start in range(0, len(data), batch_size):
        batch_images = []
        batch_class_labels = []
        batch_coord_labels = []
        for index in range(batch_start, min(batch_start + batch_size, len(data))):
            generated_image, shapes_info = generator.generate_image()
            if generated_image is not None:
                image = generated_image
                for shape_info in shapes_info:
                    class_label = shape_class_mapping[shape_info['name']]
                    class_labels_one_hot = np.zeros((num_classes,))
                    class_labels_one_hot[class_label - 1] = 1
                    coord_labels = np.array([shape_info['region']['origin']['x'],
                                             shape_info['region']['origin']['y'],
                                             shape_info['region']['size']['width'],
                                             shape_info['region']['size']['height']])
                    batch_images.append(image)
                    batch_class_labels.append(class_labels_one_hot)
                    batch_coord_labels.append(coord_labels)
    
        batch_images = np.array(batch_images)
        batch_class_labels = np.array(batch_class_labels)
        batch_coord_labels = np.array(batch_coord_labels)
        class_labels_one_hot = np.array(batch_class_labels)
        coord_labels = np.array(batch_coord_labels)
        loss = model.train_on_batch({'input': batch_images}, {'class_output': batch_class_labels, 'coord_output': batch_coord_labels})
        epoch_losses.append(loss)
    
        average_loss = np.mean(epoch_losses)
        print(f'Average Loss for epoch {epoch}: {average_loss}')

model.save('shape_detection_model.h5')

plt.plot(epoch_losses, label='Loss')
#plt.plot(epoch_accuracies, label='Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()
plt.show()