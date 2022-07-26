import argparse
from models import build_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample of TensorRT Inference.')
    parser.add_argument('arch', type=str, help='model arch')
    parser.add_argument('config', type=str, help='config file')
    parser.add_argument('folder', type=str, help='folder name')
    args = parser.parse_args()

    model = build_model(args)
    model.load_engine()
    model.inference_folder(args.folder)
    model.cfx.pop()
