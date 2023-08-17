#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:51:51 2023

@author: energia
"""

import cv2
import numpy as np
import json
import os
import random
import math


class ImageGenerator:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.image_counter = 0
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def check_intersection(self, rect1, rect2):
        x1_1, y1_1, x2_1, y2_1 = rect1
        x1_2, y1_2, x2_2, y2_2 = rect2

        return (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)

    def generate_image(self):
        image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        num_shapes = random.randint(1, 5)
        shapes_info = []
        bounding_box_coords = []

        
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'triangle', 'rectangle', 'hexagon', 'diamond'])
            color = [random.randint(0, 255) for _ in range(3)]
            intersection = False
            id = 0
            if shape_type == 'circle':
                color = [random.randint(0, 255) for _ in range(3)]
                while intersection == False:
                    r = random.randint(13, 50)
                    d = 2 * r
                    x_center = random.randint(0, 256 - 2*d) + d
                    y_center = random.randint(0, 256 - 2*d) + d
                    new_bounding_box = (x_center - r, y_center - r, x_center + r, y_center + r)
                    no_intersection = all(self.check_intersection(existing_rect, new_bounding_box) for existing_rect in bounding_box_coords)
                    if no_intersection:
                        cv2.circle(image, (x_center, y_center), r, color, -1)
                        cv2.rectangle(image, (x_center - r, y_center - r), (x_center + r, y_center + r), color=color, thickness=2)
                        bounding_box_coords.append(new_bounding_box)
                        id = 1
                        intersection = True
            
            elif shape_type == 'triangle':
                color = [random.randint(0, 255) for _ in range(3)]
                while intersection == False:
                    x1, y1 = random.randint(80, 254), random.randint(80, 254)
                    x2, y2 = random.randint(x1 - 75, x1 + 75), random.randint(y1 - 75, y1 + 75)
                    x3, y3 = random.randint(min(x1, x2), max(x1, x2)), random.randint(min(y1, y2), max(y1, y2))                        
                    triangle_points = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.int32)
                    
                    new_bounding_box = (min(x1, x2, x3), min(y1, y2, y3), max(x1, x2, x3), max(y1, y2, y3))
                    no_intersection = all(self.check_intersection(existing_rect, new_bounding_box) for existing_rect in bounding_box_coords)
                    if no_intersection:
                        cv2.polylines(image, [triangle_points], isClosed=True, color=color, thickness=2)
                        cv2.rectangle(image, (new_bounding_box[0], new_bounding_box[1]), (new_bounding_box[2], new_bounding_box[3]), color=(0, 255, 0), thickness=2)
                        bounding_box_coords.append(new_bounding_box)
                        id = 2
                        intersection = True

            elif shape_type == 'rectangle':
                color = [random.randint(0, 255) for _ in range(3)]
                while not intersection:
                    x, y = random.randint(4, 256 - 75), random.randint(4, 256 - 75)
                    w, h = random.randint(12, 75), random.randint(12, 75)
                    new_bounding_box = (x, y, x + w, y + h)
                    no_intersection = all(self.check_intersection(existing_rect, new_bounding_box) for existing_rect in bounding_box_coords)
                    if no_intersection:
                        cv2.rectangle(image, (x, y), (x+w, y+h), color=color, thickness=2)
                        bounding_box_coords.append(new_bounding_box)
                        id = 3
                        intersection = True
            
            elif shape_type == 'diamond':
                color = [random.randint(0, 255) for _ in range(3)]
                while not intersection:
                    x_center = random.randint(75, 256 - 75)
                    y_center = random.randint(75, 256 - 75)
                    side_length = random.randint(12, 60)
                    diamond_points = np.array([
                        [x_center, y_center - side_length],
                        [x_center + side_length, y_center],
                        [x_center, y_center + side_length],
                        [x_center - side_length, y_center]
                    ], dtype=np.int32)
                    x_min = np.min(diamond_points[:, 0])
                    x_max = np.max(diamond_points[:, 0])
                    y_min = np.min(diamond_points[:, 1])
                    y_max = np.max(diamond_points[:, 1])
                    new_bounding_box = (x_min, y_min, x_max, y_max)
                    no_intersection = all(self.check_intersection(existing_rect, new_bounding_box) for existing_rect in bounding_box_coords)
                    if no_intersection:
                        cv2.polylines(image, [diamond_points], isClosed=True, color=color, thickness=2)
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=2)
                        bounding_box_coords.append(new_bounding_box)
                        id = 4
                        intersection = True

            else:
            #elif shape_type == 'hexagon':
                color = [random.randint(0, 255) for _ in range(3)]
                while not intersection:
                    x_center = random.randint(75, 256 - 75)
                    y_center = random.randint(75, 256 - 75)
                    radius = random.randint(12, 50)
                    # Вычисление координат вершин гексагона на основе центральной точки и радиуса
                    hexagon_points = []
                    for i in range(6):
                        angle = math.radians(60 * i)  # Угол поворота для каждой вершины
                        x = x_center + int(radius * math.cos(angle))
                        y = y_center + int(radius * math.sin(angle))
                        hexagon_points.append((x, y))
               
                    hexagon_points = np.array(hexagon_points, dtype=np.int32)
                    x_min = np.min(hexagon_points[:, 0])
                    x_max = np.max(hexagon_points[:, 0])
                    y_min = np.min(hexagon_points[:, 1])
                    y_max = np.max(hexagon_points[:, 1])
                    new_bounding_box = (x_min, y_min, x_max, y_max)
                    no_intersection = all(self.check_intersection(existing_rect, new_bounding_box) for existing_rect in bounding_box_coords)
                    if no_intersection:
                        cv2.polylines(image, [hexagon_points], isClosed=True, color=color, thickness=2)
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=2)
                        bounding_box_coords.append(new_bounding_box)
                        id = 5
                        intersection = True                            
            
            shape_info = {
                "id": id,
                "name": shape_type,
                "region": {
                    "origin": {
                        "x": int(new_bounding_box[0]),
                        "y": int(new_bounding_box[1])
                        },
                    "size": {
                        "width": int(new_bounding_box[2] - new_bounding_box[0]),
                        "height": int(new_bounding_box[3] - new_bounding_box[1])
                        }
                    }
                }
            shapes_info.append(shape_info)

        image_path = os.path.join(self.output_folder, f'image_{self.image_counter}.png')
        json_info = os.path.join(self.output_folder, f'image_{self.image_counter}.json')
        self.image_counter += 1
        cv2.imwrite(image_path, image)

        print(f"Shape type: {shape_type}")
        print(f"New bounding box: {new_bounding_box}")
        print(f"Existing bounding boxes: {bounding_box_coords}")
        
        with open(json_info, 'w') as json_file:
            json.dump(shapes_info, json_file, indent=4)
            
if __name__ == '__main__':
    generator = ImageGenerator(output_folder='generated_images')
    for _ in range(100):
        generator.generate_image()