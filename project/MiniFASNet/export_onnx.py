import onnx
import torch
from src.anti_spoof_predict import AntiSpoofPredict
import argparse
from onnxsim import simplify


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_file', type=str, default='./resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth',
                        help='weights file path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[80, 80], help='image size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    opt = parser.parse_args()

    model_test = AntiSpoofPredict(0)
    model_test._load_model(opt.weights_file)
    model_test.model = model_test.model.cpu()
    model_test.model.eval()
    output_path = opt.weights_file.replace('pth', 'onnx')
    input_shape = opt.img_size
    dummy_input = torch.autograd.Variable(torch.randn(opt.batch_size, 3, input_shape[0], input_shape[1]))
    torch.onnx.export(model_test.model, dummy_input, output_path, keep_initializers_as_inputs=True)
    onnx_model = onnx.load(output_path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    print('ONNX export success, saved as %s' % output_path)
