import os
import tensorrt as trt
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Model:
    def __init__(self, config):
        super().__init__()
        self.onnx_file = config['onnx_file']
        self.engine_file = config['engine_file']
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.INPUT_CHANNEL = config['INPUT_CHANNEL']
        self.IMAGE_WIDTH = config['IMAGE_WIDTH']
        self.IMAGE_HEIGHT = config['IMAGE_HEIGHT']
        self.image_order = config['image_order']
        self.channel_order = config['channel_order']
        self.img_mean = config['img_mean']
        self.img_std = config['img_std']
        self.alpha = config['alpha']
        self.resize = config['resize']
        self.TRT_LOGGER = trt.Logger()
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        self.cfx = cuda.Device(0).make_context()

    def allocate_buffers(self):
        self.stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

    def load_engine(self):
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

        def build_engine():
            """Takes an ONNX file and creates a TensorRT engine to run inference with"""
            with trt.Builder(self.TRT_LOGGER) as builder, builder.create_network(
                    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                    builder.create_builder_config() as config, \
                    trt.OnnxParser(network, self.TRT_LOGGER) as parser, \
                    trt.Runtime(self.TRT_LOGGER) as runtime:
                config.max_workspace_size = 1 << 33
                builder.max_batch_size = self.BATCH_SIZE
                config.set_flag(trt.BuilderFlag.FP16)
                # Parse model file
                if not os.path.exists(self.onnx_file):
                    print('ONNX file {} not found, please run export_onnx.py first to generate it.'.format(
                        self.onnx_file))
                    exit(0)
                print('Loading ONNX file from path {}...'.format(self.onnx_file))
                with open(self.onnx_file, 'rb') as model:
                    print('Beginning ONNX file parsing')
                    parser.parse(model.read())
                print('Completed parsing of ONNX file')
                print('Building an engine from file {}; this may take a while...'.format(self.onnx_file))
                network.get_input(0).shape = [self.BATCH_SIZE, self.INPUT_CHANNEL,
                                              self.IMAGE_HEIGHT, self.IMAGE_WIDTH]
                plan = builder.build_serialized_network(network, config)
                self.engine = runtime.deserialize_cuda_engine(plan)
                print("Completed creating Engine")
                with open(self.engine_file, "wb") as f:
                    f.write(self.engine.serialize())
                return self.engine

        if os.path.exists(self.engine_file):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(self.engine_file))
            with open(self.engine_file, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        else:
            build_engine()
        self.context = self.engine.create_execution_context()
        self.allocate_buffers()

    def pre_process(self, image_list):
        image_batch = []
        for src_img in image_list:
            if src_img is None:
                continue
            flt_img = None
            if self.INPUT_CHANNEL == 1:
                flt_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
            elif self.INPUT_CHANNEL == 3:
                flt_img = src_img.copy()
            if self.resize == "directly":
                flt_img = cv2.resize(flt_img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)).astype(np.float32)
            elif self.resize == "keep_ratio":
                height, width = src_img.shape[:2]
                ratio = min(self.IMAGE_WIDTH / width, self.IMAGE_HEIGHT / height)
                flt_img = np.zeros((self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.INPUT_CHANNEL))
                rsz_img = cv2.resize(src_img, (int(width * ratio), int(height * ratio)))
                flt_img[:int(height * ratio), :int(width * ratio)] = rsz_img
            flt_img /= self.alpha
            image_batch.append(flt_img)
        image_batch = np.array(image_batch)
        if self.image_order == "BCHW":
            image_batch = np.transpose(image_batch, [0, 3, 1, 2])
        image_batch = np.array(image_batch, dtype=np.float32, order='C')
        return image_batch

    def model_inference(self, image_data):
        self.cfx.push()
        self.inputs[0].host = image_data
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        self.cfx.pop()
        return [out.host for out in self.outputs]
