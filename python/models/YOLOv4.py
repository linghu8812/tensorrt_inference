import numpy as np
import torch
from torchvision.ops import nms
from .Detection import Detection, DetectRes, Bbox


class Yolov4(Detection):
    def __init__(self, config):
        super().__init__(config)
        self.anchors = config['anchors']
        self.grids, self.refer_rows, self.refer_cols = [], 0, 6
        for index, stride in enumerate(self.strides):
            self.grids.append([self.num_anchors[index], self.IMAGE_HEIGHT // stride, self.IMAGE_WIDTH // stride])
            self.refer_rows += self.num_anchors[index] * self.IMAGE_HEIGHT // stride * self.IMAGE_WIDTH // stride
        self.refer_matrix = np.zeros((self.refer_rows, self.refer_cols))
        self.generate_refer_matrix()
        self.letter_box = self.resize == "keep_ratio"
        self.output_shape = (self.BATCH_SIZE, self.num_rows, self.CATEGORY + 5)

    def generate_refer_matrix(self):
        position = 0
        for n in range(len(self.grids)):
            for c in range(self.grids[n][0]):
                anchor = self.anchors[n * self.grids[n][0] + c]
                for h in range(self.grids[n][1]):
                    for w in range(self.grids[n][2]):
                        row = self.refer_matrix[position]
                        row[0] = w
                        row[1] = self.grids[n][2]
                        row[2] = h
                        row[3] = self.grids[n][1]
                        row[4] = anchor[0]
                        row[5] = anchor[1]
                        position += 1

    def parse_boxes(self, x, anchors):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = torch.from_numpy(x.copy())
        y[:, 0] = (y[:, 0] * 2 - 0.5 + anchors[:, 0]) / anchors[:, 1] * self.IMAGE_WIDTH if self.letter_box else \
            (torch.sigmoid(y[:, 0]) + anchors[:, 0]) / anchors[:, 1]
        y[:, 1] = (y[:, 1] * 2 - 0.5 + anchors[:, 2]) / anchors[:, 3] * self.IMAGE_HEIGHT if self.letter_box else \
            (torch.sigmoid(y[:, 1]) + anchors[:, 2]) / anchors[:, 3]
        y[:, 2] = torch.pow(y[:, 2] * 2, 2) * anchors[:, 4] if self.letter_box else \
            (torch.exp(y[:, 2]) * anchors[:, 4]) / self.IMAGE_WIDTH
        y[:, 3] = torch.pow(y[:, 3] * 2, 2) * anchors[:, 5] if self.letter_box else \
            (torch.exp(y[:, 3]) * anchors[:, 5]) / self.IMAGE_HEIGHT
        return y

    def post_process(self, image_batch, outputs):
        vec_result = []
        max_wh = 7680
        outputs = outputs[0].reshape(self.output_shape)
        for src_img, output in zip(image_batch, outputs):
            result = DetectRes()
            height, width = src_img.shape[:2]
            ratio = max(width / self.IMAGE_WIDTH, height / self.IMAGE_HEIGHT)
            if not self.letter_box:
                output[:, 4:] = torch.sigmoid(torch.from_numpy(output[:, 4:])).numpy()
            index = output[:, 4] > self.obj_threshold
            conf, j = torch.from_numpy(output[index, 5:]).max(dim=1)
            conf *= output[index, 4]
            boxes = self.parse_boxes(output[index, :4], torch.from_numpy(self.refer_matrix[index]))
            if self.letter_box:
                boxes = boxes * ratio
            else:
                boxes[:, 0::2], boxes[:, 1::2] = boxes[:, 0::2] * width, boxes[:, 1::2] * height
            boxes = self.xywh2xyxy(boxes)
            c = j * (0 if self.agnostic else max_wh)  # classes
            output_indexes = nms(boxes + c.unsqueeze(1), conf, self.nms_threshold)  # NMS
            for output_index in output_indexes:
                bbox = Bbox(j[output_index], conf[output_index], boxes[output_index, 0], boxes[output_index, 1],
                            boxes[output_index, 2] - boxes[output_index, 0],
                            boxes[output_index, 3] - boxes[output_index, 1])
                result.det_results.append(bbox)
            vec_result.append(result)
        return vec_result
