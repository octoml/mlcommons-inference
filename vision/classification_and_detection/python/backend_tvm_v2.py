"""
TVM backend for MLPerf inference vision benchmark

Developers: Alexander Peskov, Thierry Moreau, Grigori Fursin
"""

import backend

import tvm
from tvm import auto_scheduler

import numpy as np

import re
import os

from tvm.contrib.async_launcher import AsyncGraphExecutor

g_graph = None


class BackendTVM(backend.Backend):
    def __init__(self):
        super(BackendTVM, self).__init__()
        self.lib = None
        self.graph = None
        self.executor_type = None
        self.max_batchsize = None

    def version(self):
        return "N/A : TODO"

    def name(self):
        """Name of the runtime."""
        return "tvm"

    def image_format(self):
        """Requested image_format. Use a more popular layout NCHW"""
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        # Check inputs and outputs
        # Normally should be specified by MLPerf, by the command line
        # By default taken from CK packages meta to ensure reproducibility and extensibility
        x = os.environ.get('ML_MODEL_INPUT_LAYERS','').strip()
        if x != '':
           inputs = x.split(',')

        x = os.environ.get('ML_MODEL_OUTPUT_LAYERS','').strip()
        if x != '':
           outputs = x.split(',')

        self.inputs = inputs
        self.outputs = outputs

        self.output_order=None
        tmp=os.environ.get('MLPERF_TVM_OUTPUT_ORDER','')
        if tmp!='':
            import json
            self.output_order=json.loads('['+tmp+']')

        # Detect working/tmp directory to store and retreive compiled models
        work_dir = os.environ.get('MLPERF_TMP_DIR','')
        if work_dir == '':
           work_dir = os.environ.get('CK_PROGRAM_TMP_DIR','')
        if work_dir == '':
           import tempfile
           work_dir = tempfile.gettempdir()
        if work_dir == '':
           work_dir = '/tmp'

        # Check if load precompiled model
        compiled_model = os.path.join(work_dir, 'model-tvm.so')
        if model_path.endswith('.so') or model_path.endswith('.dylib'):
           compiled_model = model_path

           if not os.path.isfile(compiled_model):
               print ('')
               raise Exception("Error: Model file {} not found!".format(compiled_model))

        if os.environ.get('MLPERF_DELETE_COMPILED_MODEL','').strip().lower()=='yes' and \
           os.path.isfile(compiled_model):
              os.remove(compiled_model)

        # TODO(@peskov): who specify that?? Only outside? Looks like TVM specific WA
        # Max batch size should be passed from main function
        max_batchsize = int(os.environ.get('TVM_BATCH_SIZE','1'))

        # Select target (default: cpu)
        # TBD(@gfursin): need to provide better customization
        # of a target via external variables that can be passed
        # from CK workflows
        if os.environ.get('MLPERF_DEVICE','')=='gpu':
           ctx = tvm.cuda(0)
        else:
           ctx = tvm.cpu(0)

        # If precompiled model found, load it directly
        if os.path.isfile(compiled_model):
           print ('TVM: loading model '+compiled_model)
           self.lib = tvm.runtime.load_module(compiled_model)

        else:
           ############################################################################
           # Import model to TVM
           from tvm import relay

           input_shapes = os.environ.get('ML_MODEL_INPUT_SHAPES','').strip()
           if input_shapes == '':
               print ('')
               raise Exception("Error: ML_MODEL_INPUT_SHAPES environment variable is not defined!")

           input_shapes = input_shapes.replace('BATCH_SIZE', str(max_batchsize))

           print ('TVM model: '+model_path)

           build_conf = {}
           params = {}

           if model_path.endswith('.pt'):
              import torch
              from tvm.relay.build_module import bind_params_by_name

              shape_list = eval('[' + input_shapes + ']')

              print ('TVM shape list: '+str(shape_list))

              pytorch_model = torch.jit.load(model_path)
              pytorch_model.eval()

              mod, params = relay.frontend.from_pytorch(pytorch_model, shape_list)

              mod["main"] = bind_params_by_name(mod["main"], params)

              # Some optimizations
              mod = relay.transform.FoldConstant()(mod)

              if os.environ.get('MLPERF_TVM_USE_DNNL','').strip().lower() == 'yes':
                 from tvm.relay.op.contrib.dnnl import partition_for_dnnl
                 from tvm.driver.tvmc.common import convert_graph_layout

                 #  move to NHWC layout, prerequisite for DNNL partitioning
                 mod = convert_graph_layout(mod, "NHWC")
                 mod = relay.transform.FoldConstant()(mod)

                 mod = partition_for_dnnl(mod)

           elif model_path.endswith('.onnx'):
              import onnx

              shape_dict = eval('{' + input_shapes + '}')

              print ('TVM shape dict: '+str(shape_dict))

              onnx_model = onnx.load(model_path)

              mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

              # Some optimizations
              mod = relay.transform.DynamicToStatic()(mod)
              #mod = relay.transform.FoldExplicitPadding()(mod)

              if os.environ.get('MLPERF_TVM_TRANSFORM_LAYOUT','').strip().lower() == 'yes':
                 kernel_layout='NHWC'

                 desired_layouts = {
                     'qnn.conv2d': [kernel_layout, 'default'],
                     'nn.conv2d': [kernel_layout, 'default'],
                     'nn.conv2d_transpose': [kernel_layout, 'default'],
                     'nn.depthwise_conv2d': [kernel_layout, 'default'],
                     'nn.conv3d': [kernel_layout, 'default'],
                     'nn.conv3d_transpose': [kernel_layout, 'default'],
                 }

                 seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                                 relay.transform.FoldConstant(),
                                                 relay.transform.ConvertLayout(desired_layouts),
                                                 ])

                 with tvm.transform.PassContext(opt_level=3):
                     mod = seq(mod)

           elif model_path.endswith('.tflite'):
              # Grigori used https://tvm.apache.org/docs/tutorials/frontend/deploy_prequantized_tflite.html

              import tflite

              shape_dict = eval('{' + input_shapes + '}')

              print ('TVM shape dict: '+str(shape_dict))

              tflite_model_buf = open(model_path, "rb").read()
              tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

              mod, params = relay.frontend.from_tflite(tflite_model, shape_dict)

           else:
              print ('')
              raise Exception("Error: model extension is not supported in TVM backend ({})!".format(model_path))

           # Build model
           # TBD! Apply autotuning history!
           opt_lvl = int(os.environ.get('MLPERF_TVM_OPT_LEVEL', 3))

           target = os.environ.get('MLPERF_TVM_TARGET', 'llvm')

           target_host=None

           # New target API
           tvm_target = tvm.target.Target(target, host=target_host)

           # Check if apply history
           tvm_history_json_file = os.environ.get('MLPERF_TVM_APPLY_HISTORY','').strip()
           if tvm_history_json_file!='':
              if not os.path.isfile(tvm_history_json_file):
                 print ('')
                 raise Exception("Error: TVM history file {} not found!".format(tvm_history_json_file))

              build_conf['relay.backend.use_auto_scheduler']=True

              with auto_scheduler.ApplyHistoryBest(tvm_history_json_file):
                 with tvm.transform.PassContext(opt_level=opt_lvl, config=build_conf):
                    self.lib=relay.build(mod, target=tvm_target, params=params)
           else:
              with tvm.transform.PassContext(opt_level=opt_lvl, config=build_conf):
                 self.lib=relay.build(mod, target=tvm_target, params=params)

           self.lib.export_library(compiled_model)

           print ('TVM compiled model: '+compiled_model)

        # Init graph
        self.graph = AsyncGraphExecutor(compiled_model)

        # TODO(@apekov): Check if provided inputs/outputs match with presented in model
        # TODO(@apekov): Is there function to get names of inputs/outputs? meanwhile fill it with fake names
        if not inputs:
            inputs = [str(idx) for idx in range(self.graph.get_num_outputs())]
        if not outputs:
            outputs = [str(idx) for idx in range(self.graph.get_num_outputs())]

        # Check executors. Need vm/vm-stateful for SSD object detection models
        self.executor_type = os.environ.get('MLPERF_TVM_EXECUTOR', 'graph')

        if self.executor_type in ("graph", "debug"):
            pass
        elif self.executor_type in ("vm", "vm-stateful"):
            raise Exception("VM mode is UNSUPPORTED ...")

        self.inputs = inputs
        self.outputs = outputs

        return self

    def predict(self, feed):
        if self.executor_type in ("vm", "vm-stateful"):
            raise Exception("VM mode is UNSUPPORTED ...")
        else:
           inputs = [None] * len(self.inputs)
           for i_name, i_data in feed.items():
               input_idx = self.inputs.index(i_name)
               inputs[input_idx] = i_data

           # Run TVM inference
           res = self.graph.infer(inputs)

           # Assume that only one output produced
           tvm_output = [res]

           return tvm_output
