import os
import argparse
import torch
import cv2
import numpy as np
import time


parser = argparse.ArgumentParser(description='mnist inference')
parser.add_argument('--weights_file', default='./mnist_net.pt', type=str, help='weights file')
parser.add_argument('--input_shape', nargs='+', default=[10, 1, 28, 28], type=int, help='input shape.')
parser.add_argument('--image_folder', default='./samples', type=str, help='image folder')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torch.load(args.weights_file)

image_list = os.listdir(args.image_folder)
image_list = sorted([os.path.join(args.image_folder, image_name) for image_name in image_list])
tensor = torch.randn(args.input_shape)
for index, image_name in enumerate(image_list):
    src_img = cv2.imread(image_name, 0) / 255
    src_img = ((src_img - 0.5) / 0.5)[np.newaxis, :, :].astype(np.float32)
    tensor[index, :] = torch.from_numpy(src_img)

with torch.no_grad():
    net.eval()
    t0 = time.time()
    output = net(tensor.to(device))
    print(f'predict in {(time.time() - t0) * 1000}ms')
    print((output.max(1, keepdim=True)[1]).cpu().numpy().squeeze())

