#include "CenterFace.h"

CenterFace::CenterFace(const YAML::Node &config) : Faces(config) {
    bbox_head = 4;
    sum_of_feature = std::accumulate(feature_sizes.begin(), feature_sizes.end(), 0);
}

void CenterFace::GenerateAnchors() {}

std::vector<FacesRes> CenterFace::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<FacesRes> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat) {
        FacesRes result;
        float *out = output + index * outSize;
        int image_width = IMAGE_WIDTH / 4, image_height = IMAGE_WIDTH / 4;
        int image_size = image_width * image_height;
        float ratio = float(src_img.cols) / float(IMAGE_WIDTH) > float(src_img.rows) / float(IMAGE_HEIGHT) ?
                      float(src_img.cols) / float(IMAGE_WIDTH) : float(src_img.rows) / float(IMAGE_HEIGHT);
        int result_cols = 1 + bbox_head + landmark_head;
        for (int item = 0; item < sum_of_feature; ++item) {
            auto *current_row = out + item * result_cols;
            if (current_row[0] > obj_threshold) {
                FaceBox headbox{};
                headbox.bbox.classes = 0;
                headbox.bbox.prob = current_row[0];

                auto *bbox = current_row + 1;
                auto *keyp = current_row + 1 + bbox_head;
                auto *mask = current_row + 1 + bbox_head + landmark_head;

                headbox.bbox.h = std::exp(bbox[0]) * 4 * ratio;
                headbox.bbox.w = std::exp(bbox[1]) * 4 * ratio;
                headbox.bbox.x = ((float) (item % image_width) + bbox[2] + 0.5f) * 4 * ratio - headbox.bbox.w / 2;
                headbox.bbox.y = ((float) (item / image_height) + bbox[3] + 0.5f) * 4 * ratio - headbox.bbox.h / 2;


                for (int i = 0; i < landmark_head / 2; i++) {
                    KeyPoint point{};
                    point.x = int(
                            headbox.bbox.x + keyp[2 * i + 1] * headbox.bbox.w);
                    point.y = int(headbox.bbox.y + keyp[2 * i] * headbox.bbox.h);
                    headbox.key_points.key_points.push_back(point);
                }

                if (detect_mask and mask[0] > mask_thresh)
                    headbox.has_mask = true;
                result.faces_results.push_back(headbox);
            }
        }
        NmsDetect(result.faces_results);
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}
