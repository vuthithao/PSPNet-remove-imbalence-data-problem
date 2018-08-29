from __future__ import print_function
import colorsys
import numpy as np
from keras.models import Model
from cityscapes_labels import trainId2label
from ade20k_labels import ade20k_id2label
from pascal_voc_labels import voc_id2label


def class_image_to_image(class_id_image, class_id_to_rgb_map):
    """Map the class image to a rgb-color image."""
    colored_image = np.zeros(
        (class_id_image.shape[0], class_id_image.shape[1], 3), np.uint8)
    for row in range(class_id_image.shape[0]):
        for col in range(class_id_image.shape[1]):
            try:
                colored_image[row, col, :] = class_id_to_rgb_map[
                    int(class_id_image[row, col])].color
            except KeyError as key_error:
                print("Warning: could not resolve classid %s" % key_error)
    return colored_image


def color_class_image(class_image, model_name):
    """Color classed depending on the model used."""
    if 'cityscapes' in model_name:
        colored_image = class_image_to_image(class_image, trainId2label)
    elif 'voc' in model_name:
        colored_image = class_image_to_image(class_image, voc_id2label)
    elif 'ade20k' in model_name:
        colored_image = class_image_to_image(class_image, ade20k_id2label)
    else:
        colored_image = add_color(class_image)
    return colored_image


def add_color(img, num_classes=19):
    h, w = img.shape
    img_color = np.zeros((h, w, 3))
    color_dict = {  "road": (128, 0, 128), #hong
                    "sidewalk": (44,35,232), #xanh duwong
                    "curb": (181,0,0), #do
                    "mycar": (127,50,127),
                    "void": (0, 0, 0)   }

    img_color[img == 0] = color_dict["road"]
    img_color[img == 1] = color_dict["sidewalk"]
    img_color[img == 2] = color_dict["curb"]
    img_color[img == 3] = color_dict["mycar"]
    img_color[img == 4] = color_dict["void"]
    # for i in range(1, 151):
    #     img_color[img == i] = to_color(i)
    # img_color[img == num_classes] = (1.0, 1.0, 1.0)
    return img_color


def to_color(category):
    """Map each category color a good distance away from each other on the HSV color space."""
    v = (category - 1) * (137.5 / 360)
    return colorsys.hsv_to_rgb(v, 1, 1)


def debug(model, data):
    """Debug model by printing the activations in each layer."""
    names = [layer.name for layer in model.layers]
    for name in names[:]:
        print_activation(model, name, data)


def print_activation(model, layer_name, data):
    """Print the activations in each layer."""
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    io = intermediate_layer_model.predict(data)
    print(layer_name, array_to_str(io))


def array_to_str(a):
    return "{} {} {} {} {}".format(a.dtype, a.shape, np.min(a),
                                   np.max(a), np.mean(a))
