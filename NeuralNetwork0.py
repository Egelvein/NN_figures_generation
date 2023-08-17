#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:31:51 2023

@author: energia
"""

import os
import cv2
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from DataGeneration1 import ImageGenerator

# Папка, в которой находятся сгенерированные изображения и JSON-файлы
image_folder = 'generated_images_2'
json_folder = 'generated_images_2'

# Создание экземпляра класса ImageGenerator
generator = ImageGenerator(output_folder='generated_images_NN')

data = []

# Загрузка изображений и JSON-файлов
for image_filename in os.listdir(image_folder):
    if image_filename.endswith('.png'):
        image_path = os.path.join(image_folder, image_filename)
        json_filename = image_filename.replace('.png', '.json')
        json_path = os.path.join(json_folder, json_filename)

        if os.path.exists(json_path):
            # Загрузка JSON-файла
            with open(json_path, 'r') as json_file:
                json_data = json.load(json_file)
            # Загрузка изображения
            image = cv2.imread(image_path)
            # Добавление данных в список
            data.append({'image': image, 'json_data': json_data})

# Создание массивов изображений и соответствующих координат
images = np.array([item['image'] for item in data])
json_coords = [item['json_data'][0]['region'] for item in data]
coords = np.array([[item['origin']['x'],
                    item['origin']['y'],
                    item['size']['width'],
                    item['size']['height']] for item in json_coords])

input_shape = (256, 256, 3)  # Размер входных изображений

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
output_shape = 4  # Количество координат прямоугольника (x, y, width, height)
model.add(Dense(output_shape, activation='linear'))  # Линейная активация для координат

model.compile(loss='mean_squared_error',  # Функция потерь для регрессии
              optimizer='adam',  # Оптимизатор
              metrics=['mae'])  # Метрика: средняя абсолютная ошибка

epochs = 1000
for epoch in range(epochs):
    print(f'epoch: ', epoch)
    generated_image, shapes_info = generator.generate_image()  # Генерация нового изображения и координат
    if generated_image is not None:
        image = np.expand_dims(generated_image, axis=0)  # Добавляем размерность для подхода к модели
        json_coords = [item['region'] for item in shapes_info]
        coords = np.array([[item['origin']['x'],
                            item['origin']['y'],
                            item['size']['width'],
                            item['size']['height']] for item in json_coords])
        coords = np.expand_dims(coords, axis=0)  # Добавляем дополнительное измерение
        model.fit(image, coords, batch_size=1, epochs=8)  # Обучение модели

# Обучение модели на всех данных
history = model.fit(images, coords, batch_size=16, epochs=10, validation_split=0.2)
model.save('shape_detection_model.h5')

# Визуализация графика функции потерь
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()