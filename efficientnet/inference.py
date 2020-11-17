import os
import numpy as np
import efficientnet.tfkeras as efn
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from efficientnet.preprocessing import center_crop_and_resize
from skimage.io import imread

model = efn.EfficientNetB0(weights='imagenet')
with open('./label.txt', 'r') as f:
    text_labels = [''.join(l.split("'")[1]) for l in f]

image_list = os.listdir('./samples')
for image_name in image_list:
    image_path = os.path.join('./samples', image_name)
    print(image_path)
    image = imread(image_path)
    image_size = model.input_shape[1]
    x = center_crop_and_resize(image, image_size=image_size)
    x = preprocess_input(x, mode='torch')
    inputs = np.expand_dims(x, 0)
    expected = model.predict(inputs)
    result = decode_predictions(expected, top=1)
    print('With prob = %.2f, it contains %s' % (
        result[0][0][2] * 100, result[0][0][1]))
