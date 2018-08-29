from os import path
from os.path import join
from scipy.misc import imresize
from python_utils.preprocessing import data_generator_s31
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Reshape, Flatten
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
from keras.models import model_from_json
from python_utils.callbacks import callbacks
from keras.models import load_model
import layers_builder as layers
from keras.optimizers import SGD
from keras.models import Model
from layers_builder import Interp
import numpy as np
import argparse
import os

from keras.models import Sequential


def set_npy_weights(weights_path, model):
    npy_weights_path = join("weights", "npy", weights_path + ".npy")
    json_path = join("weights", "keras", weights_path + ".json")
    h5_path = join("weights", "keras", weights_path + ".h5")

    print("Importing weights from %s" % npy_weights_path)
    weights = np.load(npy_weights_path).item()

    for layer in model.layers:
        print(layer.name)
        if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
            mean = weights[layer.name]['mean'].reshape(-1)
            variance = weights[layer.name]['variance'].reshape(-1)
            scale = weights[layer.name]['scale'].reshape(-1)
            offset = weights[layer.name]['offset'].reshape(-1)

            model.get_layer(layer.name).set_weights(
                [scale, offset, mean, variance])

        elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
            try:
                weight = weights[layer.name]['weights']
                model.get_layer(layer.name).set_weights([weight])
            except Exception as err:
                try:
                    biases = weights[layer.name]['biases']
                    model.get_layer(layer.name).set_weights([weight,
                                                             biases])
                except Exception as err2:
                    print(err2)

        if layer.name == 'activation_52':
            break


def train(datadir, logdir, input_size, nb_classes, resnet_layers, batchsize, weights, initial_epoch, pre_trained, sep,
          eye):
    if args.weights:
        print("Loanding model...")
        json_path = join("weights", "keras", weights + ".json")
        h5_path = join("weights", "keras", weights + ".h5")
        with open(json_path, 'r') as file_handle:
            model = model_from_json(file_handle.read())
        model.load_weights(h5_path)
        if args.eye == 0:
            x = model.get_layer("dropout_1").output

            # x = Conv2D(128, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
            #            use_bias=False)(x)
            # x = layers.BN(name="conv5_4_bn")(x)
            # x = Activation('relu',name="activation_108")(x)
            # x = Dropout(0.1)(x)

            x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
            x = Interp([input_size[0], input_size[1]])(x)
            x = Activation('softmax', name="activation_110")(x)
            new_model = Model(inputs=model.input, outputs=x)
            # print(new_model.summary())
            # Solver
            # for layer in new_model.layers[:382]:
            #     layer.trainable = False
            print(new_model.summary())
        else:
            ### Can delete when have real retrain model
            # x = model.get_layer("dropout_1").output
            # x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
            # x = Interp([input_size[0], input_size[1]])(x)
            # x = Activation('softmax', name="activation_110")(x)
            # new_model_1 = Model(inputs=model.input, outputs=x)
            # ###
            new_model = layers.build_pspnet(nb_classes=nb_classes,
                                            resnet_layers=resnet_layers,
                                            input_shape=input_size)
            print(new_model.summary())
            # new_model_2.set_weights(new_model_1.get_weights())

            # x = new_model_2.get_layer("activation_58").output
            # x = Flatten()(x)
            # x = Reshape((14641, 5), input_shape=(73205,))(x)
            # # x = Activation('softmax', name="activation_110")(x)
            # new_model = Model(inputs=new_model_2.input, outputs=x)
            # print(new_model.summary())
        sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
        new_model.compile(optimizer=sgd,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    else:
        model = layers.build_pspnet(nb_classes=nb_classes,
                                    resnet_layers=resnet_layers,
                                    input_shape=input_size)
        set_npy_weights(pre_trained, model)
    dataset_len = len(os.listdir(os.path.join(datadir, 'imgs')))
    print("Create data generator...")
    train_generator, val_generator = data_generator_s31(
        datadir=datadir, batch_size=batchsize, input_size=input_size, nb_classes=nb_classes, separator=sep)
    print("Starting fitting...")

    # class_weights = np.zeros((16, 121,121, nb_classes))
    # class_weights[:, :, :, 0] += 8.6
    # class_weights[:, :, :, 1] += 63.1
    # class_weights[:, :, :, 2] += 268.
    # class_weights[:, :, :, 3] += 63.1
    # class_weights[:, :, :, 4] += 1.

    class_weights = np.zeros((14641, nb_classes))
    class_weights[:, 0] += 8.6
    class_weights[:, 1] += 63.1
    class_weights[:, 2] += 268.
    class_weights[:, 3] += 63.1
    class_weights[:, 4] += 1.

    new_model.fit_generator(
        generator=train_generator,
        epochs=100, verbose=True, steps_per_epoch=100,
        validation_data=val_generator, validation_steps=100,
        callbacks=callbacks(logdir),
        initial_epoch=initial_epoch,
        class_weight=class_weights)

    print("I am here...")


class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017"""

    def __init__(self, nb_classes, resnet_layers, input_shape):
        self.input_shape = input_shape
        self.model = layers.build_pspnet(nb_classes=nb_classes,
                                         layers=resnet_layers,
                                         input_shape=self.input_shape)
        print("Load pre-trained weights")
        self.model.load_weights("weights/keras/pspnet50_ade20k.h5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=473)
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--resnet_layers', type=int, default=50)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--initial_epoch', type=int, default=0)
    parser.add_argument('-m', '--model', type=str, default='pspnet50_ade20k',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012',
                                 'pspnet101_retrain'])
    parser.add_argument('--eye', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sep', default=').')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Starting train...")
    train(args.datadir, args.logdir, (121, 121), args.classes, args.resnet_layers,
          args.batch, args.weights, args.initial_epoch, args.model, args.sep, args.eye)
