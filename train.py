import tensorflow as tf
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import random
from src.models import get_siamese_net
from src.data_generator import DataGenerator
from src.callbacks import LRPrinter, CustomModelCheckpoint
from argparse import ArgumentParser

TRAIN_TEST_SPLIT = 0.8
IMG_SIZE = (102, 102)
LR = 0.01


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('backbone')
    parser.add_argument('scan',
                        help='path to scan')
    parser.add_argument('img',
                        help='path to img')
    parser.add_argument('output',
                        help='path to model checkpoint')
    parser.add_argument('batch_size',
                        help='batch size',
                        type=int,
                        default=8)
    parser.add_argument('gpu',
                        default='0')
    args = parser.parse_args()
    return args


def scheduler(epoch, lr):
    if epoch == 30 or epoch == 60 or epoch == 100 or epoch == 130:
        lr = 0.1 * lr
    return lr


def main():
    # create checkpoint path if not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # get data generator
    files = os.listdir(args.img)
    random.seed(12)
    train_files = files[:int(TRAIN_TEST_SPLIT * len(files))]
    val_files = files[int(TRAIN_TEST_SPLIT * len(files)):]
    print('train dataset : {} images'.format(len(train_files)))
    print('val dataset : {} images'.format(len(val_files)))
    train_generator = DataGenerator(args.scan,
                                    args.img,
                                    train_files,
                                    IMG_SIZE,
                                    args.batch_size,
                                    filter=False,
                                    preprocess='rgb',
                                    augmentation=True)
    val_generator = DataGenerator(args.scan,
                                  args.img,
                                  val_files,
                                  IMG_SIZE,
                                  args.batch_size,
                                  filter=False,
                                  preprocess='rgb',
                                  augmentation=False)

    # get model
    siamese_net = get_siamese_net(args.backbone, IMG_SIZE)

    optimizer = tf.keras.optimizers.SGD(learning_rate=LR)
    siamese_net.compile(optimizer=optimizer,
                        loss=tf.keras.losses.BinaryCrossentropy(),
                        metrics=[tf.keras.metrics.BinaryAccuracy(name='acc')])

    # set up callbacks
    checkpoint_path = os.path.join(args.output, 'checkpoint_{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.hdf5')
    best_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                       monitor='val_acc',
                                                       save_best_only=True,
                                                       save_freq='epoch')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor='val_loss',
                                                     save_freq='epoch')
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    lr_printer = LRPrinter()

    # fit
    siamese_net.fit(train_generator, epochs=500, validation_data=val_generator, 
                    batch_size=args.batch_size, callbacks=[cp_callback, lr_scheduler, lr_printer])


if __name__ == '__main__':
    args = parse_args()
    main()
