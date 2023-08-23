import torch
from torchvision.ops import nms
from .Detection import Detection, DetectRes, Bbox


class Yolov8(Detection):
    def __init__(self, config):
        super().__init__(config)
        self.output_shape = (self.BATCH_SIZE, self.CATEGORY + 4, self.num_rows)

    def post_process(self, image_batch, outputs):
        vec_result = []
        max_wh = 7680
        outputs = outputs[0].reshape(self.output_shape)
        for src_img, output in zip(image_batch, outputs):
            result = DetectRes()
            height, width = src_img.shape[:2]
            ratio = max(width / self.IMAGE_WIDTH, height / self.IMAGE_HEIGHT)
            conf, j = torch.from_numpy(output[4:, :]).max(0)
            index = conf > self.obj_threshold
            conf, j = conf[index], j[index]
            boxes = self.xywh2xyxy(output[:4, index].T) * ratio
            c = j * (0 if self.agnostic else max_wh)  # classes
            output_indexes = nms(torch.from_numpy(boxes) + c.unsqueeze(1), conf, self.nms_threshold)  # NMS
            for output_index in output_indexes:
                bbox = Bbox(j[output_index], conf[output_index], boxes[output_index, 0], boxes[output_index, 1],
                            boxes[output_index, 2] - boxes[output_index, 0],
                            boxes[output_index, 3] - boxes[output_index, 1])
                result.det_results.append(bbox)
            vec_result.append(result)
        return vec_result
