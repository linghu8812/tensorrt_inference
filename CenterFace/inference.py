import os
import cv2
from centerface import CenterFace


image_list = os.listdir('./samples')
for image_name in image_list:
    image_path = os.path.join('./samples', image_name)
    print(image_path)
    src_img = cv2.imread(image_path)
    h, w = src_img.shape[:2]
    landmarks = True
    centerface = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerface(src_img, h, w, threshold=0.35)
    else:
        dets = centerface(src_img, threshold=0.35)

    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.putText(src_img, f'{score:.2f}', (int(boxes[0]), int(boxes[1]) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (255, 0, 0), 2)
        cv2.rectangle(src_img, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (255, 0, 0), 2)
    if landmarks:
        for lm in lms:
            for i in range(0, 5):
                color = (0, 255, 0) if i % 3 == 0 else (0, 255, 255) if i % 3 == 2 else (0, 0, 255)
                cv2.circle(src_img, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, color, -1)
    print('writing', image_path[:-4] + '_.jpg')
    cv2.imwrite(image_path[:-4] + '_.jpg', src_img)
