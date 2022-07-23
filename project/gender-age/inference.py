import os
import face_model
import argparse
import cv2

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--image_folder', default='./samples/', help='')
parser.add_argument('--model', default='model/gender-age,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
args = parser.parse_args()

model = face_model.FaceModel(args)
image_names = os.listdir(args.image_folder)
image_names = sorted([os.path.join(args.image_folder, image_name) for image_name in image_names])
for image_name in image_names:
    img = cv2.imread(image_name)
    img = model.get_input(img)
    gender, age = model.get_ga(img)
    print(f'image name {image_name.split("/")[-1]}, gender is {gender}, age is {age}!')
