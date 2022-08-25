import os
import glob
import numpy as np
import cv2
import torch

from .Model import Model
from .Classification import ClassRes
from .common import read_class_label, time_sync


class Bbox(ClassRes):
    def __init__(self, classes, prob, x, y, w, h):
        super().__init__(classes, prob)
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class DetectRes:
    def __init__(self):
        super().__init__()
        self.det_results = []


class Detection(Model):
    def __init__(self, config):
        super().__init__(config)
        self.labels_file = config['labels_file']
        self.obj_threshold = config['obj_threshold']
        self.nms_threshold = config['nms_threshold']
        self.agnostic = config['agnostic']
        self.strides = config['strides']
        self.num_anchors = config['num_anchors']
        index, self.num_rows = 0, 0
        for stride in self.strides:
            num_anchor = self.num_anchors[index] if self.num_anchors[index] != 0 else 1
            self.num_rows += self.IMAGE_HEIGHT // stride * self.IMAGE_WIDTH // stride * num_anchor
            index += 1
        self.class_labels = read_class_label(self.labels_file)
        self.CATEGORY = len(self.class_labels)
        self.class_colors = np.random.randint(0, 256, (self.CATEGORY, 3))

    def inference_images(self, image_batch):
        t_start_pre = time_sync()
        image_data = self.pre_process(image_batch)
        t_end_pre = time_sync()
        total_pre = t_end_pre - t_start_pre
        print(f'detection prepare image take: {total_pre * 1000} ms.')
        t_start = time_sync()
        outputs = self.model_inference(image_data)
        t_end = time_sync()
        total_inf = t_end - t_start
        print(f'detection inference take: {total_inf * 1000} ms.')
        r_start = time_sync()
        boxes = self.post_process(image_batch, outputs)
        r_end = time_sync()
        total_res = r_end - r_start
        print(f'detection postprocess take: {total_res * 1000} ms.')
        return boxes

    def inference_folder(self, folder_name):
        image_list = glob.glob(f'{folder_name}/*.*')
        image_batch = []
        image_names = []
        total_time = 0
        for index, image_name in enumerate(image_list):
            print(f'Processing {image_name} ...')
            src_img = cv2.imread(image_name)
            image_names.append(image_name)
            if self.channel_order == 'BGR':
                src_img = src_img[:, :, ::-1]
            image_batch.append(src_img)
            if len(image_batch) == self.BATCH_SIZE or index == len(image_list) - 1:
                start_time = time_sync()
                det_results = self.inference_images(image_batch)
                end_time = time_sync()
                self.draw_results(det_results, image_batch, image_names)
                total_time += end_time - start_time
                image_batch, image_names = [], []
        print(f'Average processing time is {total_time / len(image_list) * 1000} ms')

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def post_process(self, image_batch, outputs):
        pass

    def draw_results(self, det_results, image_batch, image_names):
        for det_result, src_img, image_name in zip(det_results, image_batch, image_names):
            if self.channel_order == 'BGR':
                src_img = src_img[:, :, ::-1]
            os.makedirs('results', exist_ok=True)
            rst_name = os.path.join('results', os.path.basename(image_name))
            for rect in det_result.det_results:
                name = f'{self.class_labels[rect.classes]}-{rect.prob:.2f}'
                color = (int(self.class_colors[rect.classes][0]),
                         int(self.class_colors[rect.classes][1]),
                         int(self.class_colors[rect.classes][2]))
                cv2.putText(src_img, name, (int(rect.x), int(rect.y) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            color, 2)
                cv2.rectangle(src_img, (int(rect.x), int(rect.y)), (int(rect.x + rect.w), int(rect.y + rect.h)),
                              color, 2)
            print(rst_name)
            cv2.imwrite(rst_name, src_img)
