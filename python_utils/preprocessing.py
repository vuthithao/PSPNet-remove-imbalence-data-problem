import os
import random
import numpy as np
from scipy.misc import imresize, imread
from scipy.ndimage import zoom
from collections import defaultdict

DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])


def preprocess_img(img, input_shape):
    img = imresize(img, input_shape)
    img = img - DATA_MEAN
    img = img[:, :, ::-1]
    img.astype('float32')
    return img


def update_inputs(batch_size=None, input_size=None, num_classes=None):
    return np.zeros([batch_size, input_size[0], input_size[1], 3]), \
           np.zeros([batch_size, input_size[0], input_size[1], num_classes])


def data_generator_s31(datadir='', nb_classes=None, batch_size=None, input_size=None, separator='_', test_nmb=50):
    if not os.path.exists(datadir):
        print("ERROR!The folder is not exist")
    # listdir = os.listdir(datadir)
    data = defaultdict(dict)
    # # img from cityscapes
    # image_dir = os.path.join(datadir, "imgs/cityscapes")
    # image_paths = os.listdir(image_dir)
    # for image_path in image_paths:
    #     nmb = image_path.split('.')[0]
    #     data[nmb]['image'] = image_path

    # image from apollo
    image_dir = os.path.join(datadir, "imgs/apollo")
    image_paths = os.listdir(image_dir)
    for image_path in image_paths:
        nmb = image_path.split('.')[0]
        data[nmb]['image'] = image_path

    # # image from augument
    # image_dir = os.path.join(datadir, "imgs/annotation")
    # image_paths = os.listdir(image_dir)
    # for image_path in image_paths:
    #     nmb = image_path.split('.')[0]
    #     data[nmb]['image'] = image_path

    # # anno from cityscapes
    # anno_dir = os.path.join(datadir, "maps_bordered/cityscapes")
    # anno_paths = os.listdir(anno_dir)
    # for anno_path in anno_paths:
    #     nmb = anno_path.split('.')[0]
    #     data[nmb]['anno'] = anno_path
    #
    # anno from apollo
    anno_dir = os.path.join(datadir, "maps_bordered/apollo")
    anno_paths = os.listdir(anno_dir)
    for anno_path in anno_paths:
        nmb = anno_path.split('.')[0]
        data[nmb]['anno'] = anno_path

    # # anno from augument
    # anno_dir = os.path.join(datadir, "maps_bordered/annotation")
    # anno_paths = os.listdir(anno_dir)
    # for anno_path in anno_paths:
    #     nmb = anno_path.split('.')[0]
    #     data[nmb]['anno'] = anno_path

    # # uncommend when train with eye dataset
    # # image from eye
    # image_dir = os.path.join(datadir, "imgs")
    # image_paths = os.listdir(image_dir)
    # for image_path in image_paths:
    #   nmb = image_path.split('.')[0]
    #   data[nmb]['image'] = image_path
    #
    # #anno from eye
    # anno_dir = os.path.join(datadir, "maps_bordered")
    # anno_paths = os.listdir(anno_dir)
    # for anno_path in anno_paths:
    #   nmb = anno_path.split('.')[0]
    #   data[nmb]['anno'] = anno_path

    values = list(data.values())
    random.shuffle(values)
    print(len(values))
    test_nmb = 1
    # print(test_nmb)
    return generate(values[test_nmb:], nb_classes, batch_size, input_size, datadir), \
           generate(values[:test_nmb], nb_classes, batch_size, input_size, datadir)


def generate(values, nb_classes, batch_size, input_size, datadir):
    while 1:
        random.shuffle(values)
        # print(len(values))
        images, labels = update_inputs(batch_size=batch_size,
                                       input_size=input_size, num_classes=nb_classes)
        for i, d in enumerate(values):

            if d['image'].split('_')[0] == 'a':
                image_dir = os.path.join(datadir, "imgs/apollo")
                anno_dir = os.path.join(datadir, "maps_bordered/apollo")
            elif d['image'].split('_')[0] == '20':
                image_dir = os.path.join(datadir, "imgs/annotation")
                anno_dir = os.path.join(datadir, "maps_bordered/annotation")
            else:
                image_dir = os.path.join(datadir, "imgs/cityscapes")
                anno_dir = os.path.join(datadir, "maps_bordered/cityscapes")
                # image_dir = os.path.join(datadir, "imgs")
                # anno_dir = os.path.join(datadir, "maps_bordered")

            img = imresize(imread(os.path.join(image_dir, d['image']), mode='RGB'), input_size)
            y = imread(os.path.join(anno_dir, d['anno']), mode='L')

            h, w = input_size
            y = zoom(y, (1. * h / y.shape[0], 1. * w / y.shape[1]), order=0, prefilter=False)
            # config class
            y[y == 0] = 4  # void, trainID = 4
            y[y == 90] = 0  # road, trainID = 0
            y[y == 119] = 1  # sidewalk, trainID = 1
            y[y == 33] = 2  # cub, trainID = 2
            y[y == 81] = 3  # mycar, trainID = 3

            y = (np.arange(nb_classes) == y[:, :, None]).astype('float32')
            assert y.shape[2] == nb_classes
            images[i % batch_size] = img
            labels[i % batch_size] = y
            if (i + 1) % batch_size == 0:
                # print("Shape images and labels is {} and {}".format(images.shape, labels.shape))
                yield images, labels
                images, labels = update_inputs(batch_size=batch_size,
                                               input_size=input_size, num_classes=nb_classes)
