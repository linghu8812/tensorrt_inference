import os
import argparse
import cv2
import numpy as np
from retinaface import RetinaFace

parser = argparse.ArgumentParser(description='RetinaFace inference')
parser.add_argument('--prefix', default='./model/R50', type=str, help='prefix of model')
parser.add_argument('--epoch', default=0, type=int, help='epoch of model')
parser.add_argument('--gpuid', default=0, type=int, help='gpu id running on')
parser.add_argument('--thresh', default=0.8, type=float, help='detect thresh')
parser.add_argument('--image_folder', default='./samples', type=str, help='image folder')
args = parser.parse_args()

img_size = [1024, 1980]

detector = RetinaFace(args.prefix, args.epoch, args.gpuid, 'net3')

image_list = os.listdir(args.image_folder)
image_list = sorted([os.path.join(args.image_folder, image_name) for image_name in image_list])
for index, image_name in enumerate(image_list):
    img = cv2.imread(image_name)
    print(img.shape)
    im_shape = img.shape
    target_size = img_size[0]
    max_size = img_size[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    print('im_scale', im_scale)

    scales = [im_scale]
    flip = False

    faces, landmarks = detector.detect(img, args.thresh, scales=scales, do_flip=flip)
    print(faces.shape, landmarks.shape)

    if faces is not None:
        print('find', faces.shape[0], 'faces')
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            cv2.putText(img, f'{faces[i][4]:.2f}', (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
            color = (255, 0, 0)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            if landmarks is not None:
                landmark5 = landmarks[i].astype(np.int)
                for l in range(landmark5.shape[0]):
                    color = (0, 255, 0) if l % 3 == 0 else (0, 255, 255) if l % 3 == 2 else (0, 0, 255)
                    cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

    print('writing', image_name[:-4] + '_.jpg')
    cv2.imwrite(image_name[:-4] + '_.jpg', img)
