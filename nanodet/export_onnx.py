import onnx
import torch
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from onnxsim import simplify
import argparse


def main(config, model_path, output_path, input_shape=(320, 320), batch_size=1):
    logger = Logger(-1, config.save_dir, False)
    model = build_model(config.model)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_model_weight(model, checkpoint, logger)
    dummy_input = torch.autograd.Variable(torch.randn(batch_size, 3, input_shape[0], input_shape[1]))
    torch.onnx.export(model, dummy_input, output_path, verbose=True, keep_initializers_as_inputs=True, opset_version=10)
    onnx_model = onnx.load(output_path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    print('finished exporting onnx ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='./config/nanodet-m.yml', help='config file path')
    parser.add_argument('--weights_file', type=str, default='./weights/nanodet_m_oldversion.pth', help='weights file path')
    parser.add_argument('--output_file', type=str, default='./nanodet-m.onnx', help='onnx file path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[320, 320], help='image size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    load_config(cfg, opt.cfg_file)
    main(cfg, opt.weights_file, opt.output_file, input_shape=opt.img_size, batch_size=opt.batch_size)
