"""
TVM backend with PyTorch models (https://github.com/apache/tvm)

TBD: for now hardwired for resnet50_INT8bit_quantized.pt - need to make it more universal

Developer(s): grigori@octoml.ai
"""

import backend

import torch

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.driver.tvmc.common import convert_graph_layout

import numpy as np

from threading import Lock

import re
import os

class BackendTVM(backend.Backend):
    def __init__(self):
        super(BackendTVM, self).__init__()
        self.sess = None
        self.lock = Lock()

    def version(self):
        return tvm.__version__

    def name(self):
        """Name of the runtime."""
        return "tvm-pytorch"

    def image_format(self):
        """image_format."""
        # For now force NCHW.
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""

        # First attempt to detect input and output names via ONNX run time.
        # See backend_onnxruntime.py
        #
        # Even if inputs/outputs can be defined by MLPerf
        # TVM will need extra info about shapes to be properly initialized!

        # Grigori have noticed that TVM VM produces output on SSD models
        # that is not in an order expected by MLPerf. Hence we provide
        # this variable to provide a correct output order for MLPerf
        self.output_order=None
        tmp=os.environ.get('MLPERF_TVM_OUTPUT_ORDER','')
        if tmp!='':
            import json
            self.output_order=json.loads('['+tmp+']')

        self.inputs = inputs
        self.outputs = outputs

        # Max batch size should be passed from main function
        max_batchsize = self.max_batchsize

        print ('')
        print ('TVM PyTorch: load model ...')
        print ('')

        pytorch_model = torch.jit.load(model_path)
        pytorch_model.eval()

        shape_list = [("input", [max_batchsize, 3, 224, 224])]

        mod, params = relay.frontend.from_pytorch(pytorch_model, shape_list)

        mod["main"] = bind_params_by_name(mod["main"], params)

        #  move to NHWC layout, prerequisite for DNNL partitioning
        mod = transform.FoldConstant()(mod)
        mod = convert_graph_layout(mod, "NHWC")
        mod = transform.FoldConstant()(mod)

        #######################################################################
        # Init TVM
        # TBD: add tvm platform selector
        ctx = tvm.cpu(0)
        self.tvm_ctx = ctx

        build_conf = {'relay.backend.use_auto_scheduler': False}
        opt_lvl = int(os.environ.get('MLPERF_TVM_OPT_LEVEL', 3))

        target = os.environ.get('MLPERF_TVM_TARGET', 'llvm')

        target_host=None
        params={}

        # New target API
        tvm_target = tvm.target.Target(target, host=target_host)

        print ('')
        print ('TVM: apply extra optimizations ...')
        print ('')
        # Padding optimization
        # Adds extra optimizations
#        mod =relay.transform.FoldExplicitPadding()(mod)


        print ('')
        print ('TVM: build model ...')
        print ('')

        executor=os.environ.get('MLPERF_TVM_EXECUTOR','graph')

        # Needed for prediction
        self.tvm_executor=executor

        if executor == "graph" or executor == "debug":
            from tvm.contrib import graph_executor

            lib=relay.build(mod, target=tvm_target)

            print ('')
            print ('TVM: init graph engine ...')
            print ('')

#            lib.export_library('/tmp/tvm.so')

            self.sess = graph_executor.GraphModule(lib['default'](ctx))
        elif executor == "vm":
            from tvm.runtime.vm import VirtualMachine

            # Without history
            with tvm.transform.PassContext(opt_level=opt_lvl, config=build_conf):
                vm_exec = relay.vm.compile(mod, target=tvm_target, params=params)

            r_exec = vm_exec

            print ('')
            print ('TVM: init VM ...')
            print ('')

            self.sess = VirtualMachine(r_exec, ctx)


        # For now hardwire inputs/outputs for just 1 model
        # TBD: take info from the CK package for a given model!

        self.inputs=['input']
        self.outputs=['output']

        print ('')
        print ('TVM: model ready ...')
        print ('')

        return self


    def predict(self, feed):

        sess = self.sess

#        self.lock.acquire()

        max_batch_size = self.max_batchsize
        batch_size = max_batch_size
        for iname, data in feed.items():
            batch_size = len(data)
            if batch_size < max_batch_size:
               # Fill in with the first tensor
                data_extra = np.stack([data[0]] * (max_batch_size-batch_size))
                data = np.vstack((data, data_extra))
            elif batch_size > max_batch_size:
                raise ValueError("Internal MLPerf error: dynamic batch size > max batch size")

            sess.set_input(iname, tvm.nd.array(data))

        # Run TVM inference
        sess.run()

        # Process TVM outputs
        output = []
        for i in range(sess.get_num_outputs()):
            # Take only the output of batch size for dynamic batches
            output.append(sess.get_output(i).asnumpy()[:batch_size])

        return output
