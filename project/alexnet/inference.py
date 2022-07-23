import os
from mxnet.gluon.model_zoo import vision as models
from mxnet import image
from mxnet import nd


def transform(data):
    data = data.transpose((2, 0, 1)).expand_dims(axis=0)
    rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
    rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
    return (data.astype('float32') / 255 - rgb_mean) / rgb_std


with open('./label.txt', 'r') as f:
    text_labels = [''.join(l.split("'")[1]) for l in f]

net = models.alexnet(pretrained=True)

image_list = os.listdir('./samples')
for image_name in image_list:
    image_path = os.path.join('./samples', image_name)
    print(image_path)
    x = image.imread(image_path)
    x = image.resize_short(x, 256)
    x, _ = image.center_crop(x, (224, 224))

    prob = net(transform(x)).softmax()
    idx = prob.topk(k=1)[0]
    for i in idx:
        i = int(i.asscalar())
        print('With prob = %.5f, it contains %s' % (
            prob[0, i].asscalar(), text_labels[i]))
