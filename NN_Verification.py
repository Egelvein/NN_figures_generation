#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:10:13 2023

@author: Sinyaev V.V.
"""

import os
import cv2
import json
import numpy as np
from keras.models import load_model


def draw_rectangles(image, shape_info):
    for shape in shape_info:
        class_label = shape_class_mapping[shape['id']]
        coord = shape['region']['origin']
        width = shape['region']['size']['width']
        height = shape['region']['size']['height']
        cv2.rectangle(image, (coord['x'], coord['y']), (coord['x'] + width, coord['y'] + height), (0, 0, 0), 1)
        cv2.putText(image, class_label, (coord['x'], coord['y']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image

model = load_model('adam_4(ConvMax).h5')

new_images_folder = 'generated_images_5000'
new_json_folder = 'generated_images_5000'
output_images_folder = 'generated_images_5000_NN'

shape_class_mapping = {
    1: 'circle',
    2: 'triangle',
    3: 'rectangle',
    4: 'diamond',
    5: 'hexagon'
}

class_name_to_code = {v: k for k, v in shape_class_mapping.items()}

for image_filename in os.listdir(new_images_folder):
    if image_filename.endswith('.png'):
        image_path = os.path.join(new_images_folder, image_filename)
        json_filename = image_filename.replace('.png', '.json')
        json_path = os.path.join(new_json_folder, json_filename)

        if os.path.exists(json_path):
            # Загрузка JSON-файла
            with open(json_path, 'r') as json_file:
                json_data = json.load(json_file)
            # Загрузка изображения
            image = cv2.imread(image_path)
            input_image = np.expand_dims(image, axis=0)

            predictions = model.predict(input_image)
            class_prediction = np.argmax(predictions[0]) + 1
            coord_prediction = predictions[1]

            image_with_rectangles = draw_rectangles(image.copy(), json_data)
            output_image_path = os.path.join(output_images_folder, image_filename)
            cv2.imwrite(output_image_path, image_with_rectangles)

            for shape_info in json_data:
                class_label = shape_info['name']
                class_code = class_name_to_code[class_label]
                coord_labels = np.array([shape_info['region']['origin']['x'],
                         shape_info['region']['origin']['y'],
                         shape_info['region']['size']['width'],
                         shape_info['region']['size']['height']])               
                input_image = np.expand_dims(image, axis=0)
    
                predictions = model.predict(input_image)
                class_prediction = np.argmax(predictions[0]) + 1
                coord_prediction = predictions[1]
                
                #prediction_data = coord_prediction
                #image_with_rectangles = draw_rectangles(image.copy(), prediction_data)
            
            #output_image_path = os.path.join(output_images_folder, image_filename)
            #cv2.imwrite(output_image_path, image_with_rectangles)

            print(f"Image: {image_filename}")
            print(f"True Shape: {shape_class_mapping[class_code]}")
            print(f"Predicted Shape: {shape_class_mapping[class_prediction]}")
            #print(f"True Coordinates: {coord_labels}")
            #print(f"Predicted Coordinates: {coord_prediction}")
            print("=" * 40)