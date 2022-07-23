import argparse

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Mish
from onnxsim import simplify


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov4-p5.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[896, 896], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size(1,3,320,192) iDetection

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
        if isinstance(m, models.common.Conv) and isinstance(m.act, models.common.Mish):
            m.act = Mish()  # assign activation
        if isinstance(m, models.common.BottleneckCSP) or isinstance(m, models.common.BottleneckCSP2) \
                or isinstance(m, models.common.SPPCSP):
            if isinstance(m.bn, nn.SyncBatchNorm):
                bn = nn.BatchNorm2d(m.bn.num_features, eps=m.bn.eps, momentum=m.bn.momentum)
                bn.training = False
                bn._buffers = m.bn._buffers
                bn._non_persistent_buffers_set = set()
                m.bn = bn
            if isinstance(m.act, models.common.Mish):
                m.act = Mish()  # assign activation
        # if isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)

    model.eval()
    model.model[-1].export = True  # set Detect() layer export=True
    # y = model(img)  # dry run

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, f)
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')
