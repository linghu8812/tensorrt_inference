# ArcFace MXNet Symbol=>ONNX=>TensorRT

## 1.Reference
- **arcface github:** [https://github.com/deepinsight/insightface/tree/master/recognition](https://github.com/deepinsight/insightface/tree/master/recognition)
- **arcface arxiv:** [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **get pretrained model:** [BaiduDrive](https://pan.baidu.com/s/1wuRTf2YIsKt76TxFufsRNA) and [Dropbox](https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0)


## 2.Export ONNX Model
```
python3 export_onnx.py --input_shape 4 3 112 112
```

## 3.Build arcface_trt Project
```
mkdir build && cd build
cmake ..
make -j
```

## 4.run arcface_trt
```
./arcface_trt ../config.yaml ../samples
```
**results:**

result|![image](https://user-images.githubusercontent.com/36389436/106716971-056b8180-663a-11eb-8685-68915bd0bae7.png)|![image](https://user-images.githubusercontent.com/36389436/106717145-4368a580-663a-11eb-84c9-527f3ed2d49a.png)|![image](https://user-images.githubusercontent.com/36389436/106717235-61cea100-663a-11eb-9868-ea78291ee80c.png)|![image](https://user-images.githubusercontent.com/36389436/106717324-875baa80-663a-11eb-8bed-622f9368a69c.png)|![image](https://user-images.githubusercontent.com/36389436/106717398-a22e1f00-663a-11eb-855b-c81dc81ddcb3.png)|![image](https://user-images.githubusercontent.com/36389436/106717491-c427a180-663a-11eb-92e0-7bb688f465a0.png)|![image](https://user-images.githubusercontent.com/36389436/106718026-6e072e00-663b-11eb-8555-68196790fab5.png)|![image](https://user-images.githubusercontent.com/36389436/106718143-97c05500-663b-11eb-87d0-36edf3e5dbaa.png)|![image](https://user-images.githubusercontent.com/36389436/106718188-ab6bbb80-663b-11eb-803d-f6ecd3f8556b.png)|![image](https://user-images.githubusercontent.com/36389436/106718257-c3dbd600-663b-11eb-8fa7-f085a575288e.png) 
---|---|---|---|---|---|---|---|---|---|---
![image](https://user-images.githubusercontent.com/36389436/106716971-056b8180-663a-11eb-8685-68915bd0bae7.png)|1|0.51497477|**0.83092833**|0.44836619|0.44409686|0.44004413|0.57703531|0.48046044|0.50348091|0.52596587
![image](https://user-images.githubusercontent.com/36389436/106717145-4368a580-663a-11eb-84c9-527f3ed2d49a.png)|0.51497477|1|0.5157097|0.49093315|0.48639575|0.55684233|0.41457996|0.4557389|0.45707369|0.51120299
![image](https://user-images.githubusercontent.com/36389436/106717235-61cea100-663a-11eb-9868-ea78291ee80c.png)|**0.83092833**|0.5157097|1|0.43384337|0.43466485|0.44116706|0.55737579|0.49809921|0.50180018|0.52988255
![image](https://user-images.githubusercontent.com/36389436/106717324-875baa80-663a-11eb-8bed-622f9368a69c.png)|0.44836619|0.49093315|0.43384337|1|**0.8184306**|0.52917022|0.44513768|0.51536781|0.50124043|0.56127048
![image](https://user-images.githubusercontent.com/36389436/106717398-a22e1f00-663a-11eb-855b-c81dc81ddcb3.png)|0.44409686|0.48639575|0.43466485|**0.8184306**|1|0.53311759|0.48287207|0.50482482|0.52335793|0.49513683
![image](https://user-images.githubusercontent.com/36389436/106717491-c427a180-663a-11eb-92e0-7bb688f465a0.png)|0.44004413|0.55684233|0.44116706|0.52917022|0.53311759|1|0.46499243|0.51840144|0.4833495|0.43685332
![image](https://user-images.githubusercontent.com/36389436/106718026-6e072e00-663b-11eb-8555-68196790fab5.png)|0.57703531|0.41457996|0.55737579|0.44513768|0.48287207|0.46499243|1|0.53517133|0.51514965|0.48933336
![image](https://user-images.githubusercontent.com/36389436/106718143-97c05500-663b-11eb-87d0-36edf3e5dbaa.png)|0.48046044|0.4557389|0.49809921|0.51536781|0.50482482|0.51840144|0.53517133|1|0.4795776|0.47983229
![image](https://user-images.githubusercontent.com/36389436/106718188-ab6bbb80-663b-11eb-803d-f6ecd3f8556b.png)|0.50348091|0.45707369|0.50180018|0.50124043|0.52335793|0.4833495|0.51514965|0.4795776|1|0.51290995
![image](https://user-images.githubusercontent.com/36389436/106718257-c3dbd600-663b-11eb-8fa7-f085a575288e.png)|0.52596587|0.51120299|0.52988255|0.56127048|0.49513683|0.43685332|0.48933336|0.47983229|0.51290995|1
