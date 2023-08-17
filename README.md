# NN_figures_generation

*Проект находится в ветке master.*

## Содержание
- [Описание](#описание)
- [Технологии](#технологии)
- [Системные требования](#системные-требования)
- [Deploy](#deploy)
- [Использование](#использование)
- [Ссылки](#ссылки)


## Описание
- Обучение нейронной сети на распознавание простейших геометрических фигур (круг, треугольник, прямоугольник, ромб, гексагон). Первым этапом была написана программа для генерации датасета - изображений, который содержат на себе от 1 до 5 фигур, разноцветных и непересекающихся друг с другом, а также информации (которая хранится в файлах .json) с типом фигуры и координатами ограничивающих прямоугольников (BoundingBox'ов). Эта программа лежит в каталоге проекта и называется DataGeneration1.py. 
- Вторым этапом происходит обучение нейронной сети на получившемся датасете (5000 изображений и файлов с их описанием), а также обучение с использованием класса для генерации изображений во время обучения - программы NeuralNetwork5000.py и NeuralNetwork0.py.

## Технологии
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org)

## Системные требования

## Deploy

## Использование

## Cсылки
Все данные хранятся [здесь](https://drive.google.com/drive/folders/1tVMbivYXtO3TjhztwvUc50CCZNBPsuzs?usp=sharing)
