# Необходимые зависимости

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as tfs
import shutil
from ultralytics import solutions
from ultralytics import YOLO
import torch.nn as nn
from PIL import Image
import easyocr
import re


class MainPipeline:
    """
    Класс, реализующий непосредственно пайплайн для выделения самоката на изображении,
    Определения бренда по цвету,
    Определения номера самоката
    """

    def __init__(self, input_image_path: str,
                 detection_scooter_weights: str, detection_number_weights: str):
        """
        Конструктор класса
        input_image - исходное изображение;
        detection_scooter(или number)_weights - веса для модели детекции YOLO;
        path_scooter(code)_crop - буферные директории для сохраниния промежуточных результатов 
        """
        self.input_image = input_image_path
        self.detection_scooter_weights = detection_scooter_weights
        self.detection_number_weights = detection_number_weights
        self.path_scooter_crop = ''
        self.path_code_crop = ''

    def _make_frame(self):
        """
        Читаем изображение при помощи cv2
        """
        return cv2.imread(self.input_image)
    
    def _check_digits_count(self):
        """
        Получаем прочитанные цифры и проверям наличие четырех цифр в результатх
        """
        results = self.easy_digits()
        for res in results:
            try:
                text = res[0][1]
                pattern = r'\d{4}'
                matches = re.findall(pattern, text)
                if len(matches[0]) == 4:
                    print(matches[0])
            except:
                pass

    def mask_color_create(self, frame):
        """
        Создаем цветовую маску в формате HSV с заранее установленными границами,
        Если пикселей достаточно, то класиифицируем самокат, как наш
        """
        color_pixels = dict()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_purple = np.array([115, 100, 100])
        upper_purple = np.array([150, 255, 255])

        lower_blue = np.array([104, 117, 92]) 
        upper_blue = np.array([114, 255, 255])

        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)     

        purp_pixels = cv2.countNonZero(mask_purple)
        bl_pixels = cv2.countNonZero(mask_blue)

        color_pixels['purple'] = purp_pixels
        color_pixels['blue'] = bl_pixels

        flag_blue = True if color_pixels['blue'] > 2000 else False
        flag_purple = True if color_pixels['purple'] > 2000 else False

        return flag_blue | flag_purple

    def mark_classifer(self):
        """
        Применяем детекцию, обрезаем изображения
        Ходим по директории, куда сохраняем результаты детекции и определяем фирму при помощи mask_color_create
        """
        frame = self._make_frame()
        cropper = solutions.ObjectCropper(
        model=self.detection_scooter_weights,
        crop_dir = 'cropped-scooters'
        )
        self.path_scooter_crop = cropper.crop_dir
        to_cropp = cropper(frame)
        total = dict()
        cropped_items = []
        for i in range(len(os.listdir(cropper.crop_dir))):
            cropped_item = os.path.join(cropper.crop_dir, os.listdir(cropper.crop_dir)[i])
            cropped_items.append(cropped_item)
            frame = cv2.imread(cropped_item)
            data = self.mask_color_create(frame=frame)
            total[i] = data

        return total

    def detect_number(self):
        """
        Находим номера(а) и сохраняем их в буферную директорию
        """
        total = self.mark_classifer()
        cropper = solutions.ObjectCropper(
        model=self.detection_number_weights,
        crop_dir = 'cropped_code'
        )
        self.path_code_crop = cropper.crop_dir
        check = total

        cropped_items = []
        # Цикл, в котором обрабатываются изображения из директории, куда сохраняются обрезанные изображения
        for i in range(len(os.listdir(self.path_scooter_crop))):
            if check[i]:
                cropped_item = os.path.join(self.path_scooter_crop, os.listdir(self.path_scooter_crop)[i])
                cropped_items.append(cropped_item)
                frame = cv2.imread(cropped_item)
                to_cropp = cropper(frame)

        return True

    def easy_digits(self):
        """
        Создаем объект reader из EasyOCR
        Устанавливаем символы, которые необходимо считать
        """
        reader = easyocr.Reader(["en"])
        allowlist = '0123456789'
        # Обертка, которая применит модель детекции и возьмет отдельные рамки для изображений
        # Цикл, в котором обрабатываются изображения из директории, куда сохраняются обрезанные изображения
        ress = []
        for i in range(len(os.listdir(self.path_code_crop))):
            cropped_item = os.path.join(self.path_code_crop, os.listdir(self.path_code_crop)[i])
            result = reader.readtext(cropped_item, allowlist=allowlist)
            ress.append(result)

        return ress

    def predict(self):
        """
        Выполняем функционал
        """
        self.detect_number()
        self._check_digits_count()
        self.clean_buffer()

    def clean_buffer(self):
        """
        Удаляем буферные директории
        """
        shutil.rmtree(self.path_scooter_crop)
        shutil.rmtree(self.path_code_crop)


if __name__ == '__main__':
    pipeline = MainPipeline(input_image_path='IMG_4885_jpg.rf.db98a7a6c8a3aaa6a7d1cc832a0a4af1.jpg',
                            detection_scooter_weights='last_detectio.pt',
                            detection_number_weights='best_code_detection.pt')
    pipeline.predict()
