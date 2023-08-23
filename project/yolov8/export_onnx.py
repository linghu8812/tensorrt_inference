import argparse
from ultralytics import YOLO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='yolov8 detection model')
    parser.add_argument('--image-size', type=int, default=640, help='yolov8 inference image size')
    parser.add_argument('--batch-size', type=int, default=1, help='yolov8 inference image size')
    opt = parser.parse_args()
    model = YOLO(opt.model)
    model.export(format='onnx', opset=12, imgsz=opt.image_size, batch=opt.batch_size)
