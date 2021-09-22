import tensorflow as tf
import numpy as np
import os
import math
import random
import cv2
from imgaug.augmenters.meta import OneOf
import imgaug.augmenters as iaa


ddepth = cv2.CV_16S
scale = 1
delta = 0


def sometimes(aug): return iaa.Sometimes(0.5, aug)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, scan_path, img_path, img_files, img_size, \
                 batch_size=8, preprocess='rgb', filter=False, augmentation=True):
        self.scan_path = scan_path
        self.img_path = img_path
        self.img_files = img_files
        random.shuffle(self.img_files)
        self.batch_size = batch_size
        self.img_size = img_size
        self.preprocess =preprocess
        self.augmentation = augmentation
        self.filter = filter
        if self.augmentation:
            self.seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Sometimes(0.5, iaa.GaussianBlur((0, 1.5))),  # apply Gaussian blur with a sigma between 0 and 3 to 50% of the images
                    iaa.Sometimes(0.5, iaa.Dropout((0.01, 0.20), per_channel=0.5)),
                ]),
                iaa.SomeOf((0, 3), [
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.8, 1.2)),  # sharpen images
                    iaa.GammaContrast((0.5, 2.0), per_channel=True),
                    iaa.MotionBlur(k=7)
                ])
            ],
                random_order=True  # apply the augmentations in random order
            )

    def augment(self, img):
        img = self.seq(images=[img])
        return img[0]

    def __len__(self):
        return math.ceil(len(self.img_files) / self.batch_size)

    def __getitem__(self, index):
        inputs_1 = self.img_files[index * self.batch_size:(index + 1) * self.batch_size]
        targets = np.zeros((self.batch_size,))
        targets[len(inputs_1) // 2:] = 1
        inputs_2 = random.choices(list(set(self.img_files) - set(inputs_1)), k=len(inputs_1) // 2)
        scans = inputs_1[len(inputs_1) // 2:]
        inputs_2.extend(scans)
        pairs = self.__data_generator(inputs_1, inputs_2)
        return pairs, targets

    def __data_generator(self, inputs_1, inputs_2):
        inputs_1_imgs = self.__get_img_by_name(inputs_1, self.img_path)
        inputs_2_imgs = self.__get_img_by_name(inputs_2, self.scan_path)
        return [inputs_1_imgs, inputs_2_imgs]

    def __get_img_by_name(self, files, path):
        imgs = []
        dir = path.split('/')[:-1]
        dir = '/'.join(dir)
        mask_path = os.path.join(dir, 'mask_' + path.split('/')[-1])
        # print(mask_path)
        for file in files:
            img = cv2.imread(os.path.join(path, file))
            # filter mask
            if self.filter:
                mask = cv2.imread(os.path.join(mask_path, file)) // 255
                img = img * mask

            if self.preprocess == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.augmentation:
                    img = self.augment(img)
            elif self.preprocess == 'sobel':
                if self.augmentation:
                    img = self.augment(img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3,
                                   scale=scale, delta=delta,
                                   borderType=cv2.BORDER_DEFAULT)
                grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3,
                                   scale=scale, delta=delta, 
                                   borderType=cv2.BORDER_DEFAULT)
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                img = np.stack([grad, grad, grad], axis=2)
            else:
                raise NotImplemented

            img = cv2.resize(img, self.img_size)
            imgs.append(img)
        imgs = np.asarray(imgs)
        return imgs
