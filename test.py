import tensorflow as tf
import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from src.models import DistanceLayer, get_preprocess_input
from argparse import ArgumentParser


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


IMG_SIZE = (102, 102)
THR = 0.5
ddepth = cv2.CV_16S
scale = 1
delta = 0

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('cp',
                        help='path to checkpoint')
    parser.add_argument('backbone',
                        help='name of backbone')
    parser.add_argument('scan',
                        help='path to scan dataset')
    parser.add_argument('img',
                        help='path to img dataset')
    parser.add_argument('--use-mask',
                        action='store_true',
                        help='use mask to filter')
    parser.add_argument('--edge',
                        action='store_true',
                        help='preprocess with edge detection')
    args = parser.parse_args()
    return args


def get_id(name):
    x = name.split('/')[-1]
    return x.split('.')[0]


def prepare_image(path, preprocessing, filter=False, edge=False):
    img = cv2.imread(path)
    if filter:
        mask_path = path.split('/')[:-2]
        mask_path = '/'.join(mask_path)
        name = path.split('/')[-2]
        file = path.split('/')[-1]
        mask_path = os.path.join(mask_path, 'mask_' + name, file)
        mask = cv2.imread(mask_path) // 255
        img = img * mask
    if edge:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, \
                            scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, \
                            scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        img = np.stack([grad, grad, grad], axis=2)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)
    return img


def main():
    # init model
    model = tf.keras.models.load_model(args.cp,
                                       custom_objects={'DistanceLayer': DistanceLayer})
    preprocessing = get_preprocess_input(args.backbone)

    db_files = os.listdir(args.scan)
    predictions = []
    for file in tqdm(db_files):
        scan = prepare_image(os.path.join(args.scan, file),
                             preprocessing, filter=args.use_mask,
                             edge=args.edge)    
        img = prepare_image(os.path.join(args.img, file),
                            preprocessing, filter=args.use_mask,
                            edge=args.edge)
        pred = model([img, scan])
        if pred > THR:
            predictions.append(1)
        else:
            predictions.append(0)

    ids = [get_id(x) for x in db_files]
    data = {'id': ids, 'prediction': predictions}
    df_submit = pd.DataFrame(data=data)
    df_submit[["id", "prediction"]].reset_index(drop=True)
    df_submit.to_csv("./submission.csv", index=False)
    

if __name__ == '__main__':
    args = parse_args()
    main()
