import os
import cv2
import numpy as np
import argparse
from easydict import EasyDict as edict
import mxnet as mx
from mxnet import ndarray as nd
from sklearn import preprocessing


def get_features(image_list, input_shape, model):
    data = nd.zeros(shape=input_shape)
    for index, image_name in enumerate(image_list):
        img = cv2.imread(image_name, cv2.IMREAD_COLOR)
        if not img.data:
            continue
        img = cv2.resize(img, (input_shape[2], input_shape[3]))[:, :, ::-1]  # to rgb
        data[index] = nd.array(np.transpose(img, (2, 0, 1)))
    db = mx.io.DataBatch(data=(data,))
    model.model.forward(db, is_train=False)
    x = model.model.get_outputs()[0].asnumpy()
    embedding = preprocessing.normalize(x)
    return embedding


parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default=0, type=int, help="gpu number")
parser.add_argument('--prefix', default='./model', help='prefix to load model.')
parser.add_argument('--epoch', default=0, type=int, help='epoch number to load model.')
parser.add_argument('--input_shape', nargs='+', default=[1, 3, 112, 112], type=int, help='input shape.')
parser.add_argument('--image_folder', default='./samples/', type=str, help='image folder name')
args = parser.parse_args()

net = edict()
net.ctx = mx.gpu(args.gpus)
net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
all_layers = net.sym.get_internals()
net.sym = all_layers['fc1_output']
net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names=None)
net.model.bind(data_shapes=[('data', args.input_shape)])
net.model.set_params(net.arg_params, net.aux_params)
image_names = os.listdir(args.image_folder)
image_names = sorted([os.path.join(args.image_folder, image_name) for image_name in image_names])
assert (len(image_names) == args.input_shape[0])
face_features = get_features(image_names, args.input_shape, net)
similarity = face_features.dot(face_features.transpose())
print(f'The similarity matrix of the image folder is:\n', (similarity + 1) / 2)
