# fast-reid PyTorch=>ONNX=>TensorRT

## 1.Reference
- **fast-reid github:** [https://github.com/JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid)
- **fast-reid arxiv:** [FastReID: A Pytorch Toolbox for General Instance Re-identification](https://arxiv.org/abs/2006.02631)

## 2.Export ONNX Model
```
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid/tools/deploy
python onnx_export.py --config-file root-path/bagtricks_R50/config.yml --name baseline_R50 --output outputs/onnx_model --opts MODEL.WEIGHTS root-path/logs/market1501/bagtricks_R50/model_final.pth
```

## 3.Build tensorrt_inference Project
```
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 4.run tensorrt_inference
```
cd ../../bin/
./tensorrt_inference fastreid ../configs/fast-reid/config.yaml ../samples/pedestrian_reid
```

**results:**

result|![image](https://user-images.githubusercontent.com/36389436/180164109-da6836d2-84c3-428f-8423-b86bab61832a.png)|![image](https://user-images.githubusercontent.com/36389436/180164183-c07de53e-a824-44a8-9cbc-e39efc2747f9.png)|![image](https://user-images.githubusercontent.com/36389436/180164215-1faeffdf-d349-4fed-b0d7-1ba3561eb9f7.png)|![image](https://user-images.githubusercontent.com/36389436/180164269-43ce0eb9-dbee-4197-853e-570ad69dd081.png)|![image](https://user-images.githubusercontent.com/36389436/180164309-09a597e9-f19a-41b7-a801-b068713697d1.png)|![image](https://user-images.githubusercontent.com/36389436/180164340-4a50ef2a-5a13-4dcc-8109-08ef2af2d4ac.png)|![image](https://user-images.githubusercontent.com/36389436/180164388-aa3e7c2b-3197-4d3f-a295-3f2ff950a93c.png)|![image](https://user-images.githubusercontent.com/36389436/180164420-32a558e1-6a3d-43f8-aa2e-79eabdf4a09b.png)|![image](https://user-images.githubusercontent.com/36389436/180166036-622e6c48-a663-4bb3-a01f-5ad0df079272.png)|![image](https://user-images.githubusercontent.com/36389436/180165528-46c07e48-56b6-4dd2-b8ff-3bbb4c083db1.png)
---|---|---|---|---|---|---|---|---|---|---
![image](https://user-images.githubusercontent.com/36389436/180164109-da6836d2-84c3-428f-8423-b86bab61832a.png)| 1 | 0.93663549 | 0.63691914 | 0.65447605 | 0.59582067 | 0.63569337 | 0.67968875 | 0.6704126 | 0.64113915 | 0.6652534
![image](https://user-images.githubusercontent.com/36389436/180164183-c07de53e-a824-44a8-9cbc-e39efc2747f9.png)| 0.93663549 | 1 | 0.60606658 | 0.63805777 | 0.58507961 | 0.62270737 | 0.65357053 | 0.64495277 | 0.65284139 | 0.67049634
![image](https://user-images.githubusercontent.com/36389436/180164215-1faeffdf-d349-4fed-b0d7-1ba3561eb9f7.png)| 0.63691914 | 0.60606658 | 1 | 0.85554099 | 0.64959818 | 0.65541494 | 0.63463104 | 0.58980453 | 0.64554667 | 0.63664776
![image](https://user-images.githubusercontent.com/36389436/180164269-43ce0eb9-dbee-4197-853e-570ad69dd081.png)| 0.65447605 | 0.63805777 | 0.85554099 | 1 | 0.65920269 | 0.64546573 | 0.64599162 | 0.59834015 | 0.66127765 | 0.65907228
![image](https://user-images.githubusercontent.com/36389436/180164309-09a597e9-f19a-41b7-a801-b068713697d1.png)| 0.59582067 | 0.58507961 | 0.64959818 | 0.65920269 | 1 | 0.87445605 | 0.72793484 | 0.7426371 | 0.68983912 | 0.69091672
![image](https://user-images.githubusercontent.com/36389436/180164340-4a50ef2a-5a13-4dcc-8109-08ef2af2d4ac.png)| 0.63569337 | 0.62270737 | 0.65541494 | 0.64546573 | 0.87445605 | 1 | 0.77302706 | 0.77868611 | 0.69440424 | 0.70926988
![image](https://user-images.githubusercontent.com/36389436/180164388-aa3e7c2b-3197-4d3f-a295-3f2ff950a93c.png)| 0.67968875 | 0.65357053 | 0.63463104 | 0.64599162 | 0.72793484 | 0.77302706 | 1 | 0.91233504 | 0.65651429 | 0.66451794
![image](https://user-images.githubusercontent.com/36389436/180164420-32a558e1-6a3d-43f8-aa2e-79eabdf4a09b.png)| 0.6704126 | 0.64495277 | 0.58980453 | 0.59834015 | 0.7426371 | 0.77868611 | 0.91233504 | 1 | 0.63502818 | 0.64097649
![image](https://user-images.githubusercontent.com/36389436/180166036-622e6c48-a663-4bb3-a01f-5ad0df079272.png)| 0.64113915 | 0.65284139 | 0.64554667 | 0.66127765 | 0.68983912 | 0.69440424 | 0.65651429 | 0.63502818 | 1 | 0.92471766
![image](https://user-images.githubusercontent.com/36389436/180165528-46c07e48-56b6-4dd2-b8ff-3bbb4c083db1.png)| 0.6652534 | 0.67049634 | 0.63664776 | 0.65907228 | 0.69091672 | 0.70926988 | 0.66451794 | 0.64097649 | 0.92471766 | 1
