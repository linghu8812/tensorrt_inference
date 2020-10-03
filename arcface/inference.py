import cv2
import numpy as np
import argparse
from easydict import EasyDict as edict
import mxnet as mx
from mxnet import ndarray as nd
from sklearn import preprocessing


def get_feature(image_name, input_shape, model):
    img = cv2.imread(image_name, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (input_shape[2], input_shape[3]))[:, :, ::-1]  # to rgb
    data = nd.zeros(shape=input_shape)
    data[0] = nd.array(np.transpose(img, (2, 0, 1)))
    db = mx.io.DataBatch(data=(data,))
    model.model.forward(db, is_train=False)
    x = model.model.get_outputs()[0].asnumpy()
    embedding = preprocessing.normalize(x)
    return embedding


parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default=0, type=int, help="gpu number")
parser.add_argument('--prefix', default='./model', help='prefix to load model.')
parser.add_argument('--epoch', default=0, type=int, help='epoch number to load model.')
parser.add_argument('--input-shape', default=(1, 3, 112, 112), type=tuple, help='input shape.')
parser.add_argument('--anchor_name', default='./samples/test1.jpg', type=str, help='first image name')
parser.add_argument('--test_name', default='./samples/test2.jpg', type=str, help='second image name')
args = parser.parse_args()

net = edict()
net.ctx = mx.gpu(args.gpus)
net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
all_layers = net.sym.get_internals()
net.sym = all_layers['fc1_output']
net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names=None)
net.model.bind(data_shapes=[('data', args.input_shape)])
net.model.set_params(net.arg_params, net.aux_params)
anchor_features = get_feature(args.anchor_name, args.input_shape, net)
test_features = get_feature(args.test_name, args.input_shape, net)
similarity = anchor_features.dot(test_features.transpose())
print(f'The similarity of the two images is: {float(similarity):.4f}!')
