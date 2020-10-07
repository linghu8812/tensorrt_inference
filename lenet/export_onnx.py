import onnx
import torch
import yaml
import argparse

parser = argparse.ArgumentParser(description='convert mnist models to onnx')
parser.add_argument('--config', default='config.yaml', type=str, help='config file')
args = parser.parse_args()

f = open(args.config)
config = yaml.load(f, Loader=yaml.FullLoader)['lenet']

# export from pytorch to onnx
net = torch.load(config['model_file']).to('cpu')
image = torch.randn(config['BATCH_SIZE'], config['INPUT_CHANNEL'], config['IMAGE_WIDTH'], config['IMAGE_HEIGHT'])
torch.onnx.export(net, image, config['onnx_file'], input_names=['input'], output_names=['output'])

# check onnx model
onnx_model = onnx.load(config['onnx_file'])  # load onnx model
onnx.checker.check_model(onnx_model)
