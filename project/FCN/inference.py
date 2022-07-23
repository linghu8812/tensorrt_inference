import os
import mxnet as mx
from mxnet import image
from gluoncv.data.transforms.presets.segmentation import test_transform
import gluoncv
from gluoncv.utils.viz import get_color_pallete

model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)
ctx = mx.cpu(0)

image_list = os.listdir('./samples')
for image_name in image_list:
    image_path = os.path.join('./samples', image_name)
    print(image_path)
    img = image.imread(image_path)
    img = test_transform(img, ctx)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    mask = get_color_pallete(predict, 'pascal_voc')
    mask.save(image_path.replace('jpg', 'png').replace('jpeg', 'png'))
