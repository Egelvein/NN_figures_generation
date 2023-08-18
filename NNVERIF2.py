#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:27:23 2023

@author: Sinyaev V.V.
"""

import os
import cv2
import numpy as np
from keras.models import load_model

# Загрузка модели
model = load_model('adam_4(ConvMax).h5')

# Путь к папке с новыми изображениями
new_images_folder = 'generated_images_5000'
output_images_folder = 'generated_images_5000_NNN'

shape_class_mapping = {
    1: 'circle',
    2: 'triangle',
    3: 'rectangle',
    4: 'diamond',
    5: 'hexagon'
}

# Перебор новых изображений
for image_filename in os.listdir(new_images_folder):
    if image_filename.endswith('.png'):
        image_path = os.path.join(new_images_folder, image_filename)

        image = cv2.imread(image_path)

        input_image = np.expand_dims(image, axis=0)

        predictions = model.predict(input_image)
        class_predictions = np.argmax(predictions[0], axis=1) + 1
        coord_predictions = predictions[1]

        for class_prediction, coord_prediction in zip(class_predictions, coord_predictions):
            class_label = shape_class_mapping[class_prediction]

            x, y, width, height = coord_prediction
            x = int(x)
            y = int(y)
            width = int(width)
            height = int(height)

            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 1)
            cv2.putText(image, class_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        output_image_path = os.path.join(output_images_folder, image_filename)
        cv2.imwrite(output_image_path, image)