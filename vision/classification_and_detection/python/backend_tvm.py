"""
TVM backend (https://github.com/apache/tvm)
"""

import onnx
import onnxruntime as rt

import backend

import tvm
from tvm import relay

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
        return rt.__version__

    def name(self):
        """Name of the runtime."""
        return "tvm"

    def image_format(self):
        """image_format."""
        # We use ONNX format, which is always NCHW.
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

        opt = rt.SessionOptions()
        print ('')
        print ('ONNX: initialize runtime to get some model parameters ...')
        print ('')

        tmp_sess = rt.InferenceSession(model_path, opt)

        if not inputs:
            self.inputs = [meta.name for meta in tmp_sess.get_inputs()]
        if not outputs:
            self.outputs = [meta.name for meta in tmp_sess.get_outputs()]

        # Detect shapes and set max batch size.
        # If batch size is < max batch size, fill in with empty ones
        # In the future, we should support dynamic batch sizes in TVM

        # Max batch size should be passed from main function
        max_batchsize = self.max_batchsize

        shape_dict = {}
        bsize_dict = {}
        dtype_dict = {}

        for meta in tmp_sess.get_inputs():
            input_name = meta.name
            input_type_str = meta.type
            input_shape = meta.shape

            input_type = re.search(r"\(([A-Za-z0-9_]+)\)", input_type_str).group(1)

            if input_type == 'float': 
                input_type='float32'

            dtype_dict[input_name] = input_type

            # For now, we expect that input_shape[0] == batch_size
            # We force it to 1
            input_shape[0] = 1
            shape_dict[input_name] = tuple(input_shape)

            bsize_dict[input_name] = max_batchsize

        print ('')
        print ('TVM: input shape(s): '+str(shape_dict))
        print ('TVM: input type: '+str(dtype_dict))
        print ('TVM: outputs: '+str(self.outputs))
        print ('')

        self.input_shapes = shape_dict
        self.input_batch_sizes = bsize_dict

        # We do not need ONNX runtime anymore
        del tmp_sess

        # Load model via ONNX to be used with TVM
        print ('')
        print ('ONNX: load model ...')
        print ('')

        onnx_model = onnx.load(model_path)

        # Init TVM
        # TBD: add tvm platform selector
        ctx = tvm.cpu(0)
        self.tvm_ctx = ctx

        build_conf = {'relay.backend.use_auto_scheduler': False}
        opt_lvl = int(os.environ.get('MLPERF_TVM_OPT_LEVEL', 3))

        target = os.environ.get('MLPERF_TVM_TARGET', 'llvm -mcpu=znver2')

        target_host=None
        params={}

        # New target API
        tvm_target = tvm.target.Target(target, host=target_host)

        print ('')
        print ('TVM: import model ...')
        print ('')
        # Extra param: opset=12
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

        print ('')
        print ('TVM: transform to static ...')
        print ('')
        mod = relay.transform.DynamicToStatic()(mod)

        print ('')
        print ('TVM: apply extra optimizations ...')
        print ('')
        # Padding optimization
        # Adds extra optimizations
        mod =relay.transform.FoldExplicitPadding()(mod)


        print ('')
        print ('TVM: build model ...')
        print ('')

        executor=os.environ.get('MLPERF_TVM_EXECUTOR','graph')

        # Needed for prediction
        self.tvm_executor=executor

        if executor == "graph" or executor == "debug":
            from tvm.contrib import graph_executor

            # Without history
            with tvm.transform.PassContext(opt_level=opt_lvl, config=build_conf):
                graph_module = relay.build(mod,
                                           target=tvm_target,
                                           params=params)
            lib = graph_module

            print ('')
            print ('TVM: init graph engine ...')
            print ('')

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

        print ('')
        print ('TVM: model ready ...')
        print ('')

        return self


    def predict(self, feed):
        """Run the prediction."""

        executor = self.tvm_executor

        sess = self.sess

        self.lock.acquire()

        if executor=='vm':
            input_list = []
            for iname, data in feed.items():
                input_list.append(tvm.nd.array(data, device=self.tvm_ctx))

            tvm_output = sess.invoke("main", *input_list)
            if not self.output_order:
               tvm_output = [x.asnumpy() for x in tvm_output]
            else:
               tvm_output = [tvm_output[x].asnumpy() for x in self.output_order]

        elif executor=='vm-stateful':
            input_list = []
            for iname, data in feed.items():
                input_list.append(tvm.nd.array(data, device=self.tvm_ctx))

            sess.invoke_stateful("main", *input_list)

            tvm_output = sess.get_outputs()
            if not self.output_order:
               tvm_output = [x.asnumpy() for x in tvm_output]
            else:
               tvm_output = [tvm_output[x].asnumpy() for x in self.output_order]
        else:
            # Prepare TVM inputs
            tvm_output = []

            for iname, data in feed.items():
                for d in data:
                    sess.set_input(iname, tvm.nd.array([d]))

                    # Run TVM inference
                    sess.run()

                    # Process TVM outputs
                    for i in range(sess.get_num_outputs()):
                        # Take only the output of batch size for dynamic batches
                        if len(tvm_output)<(i+1):
                            tvm_output.append([])
                        tvm_output[i].append(sess.get_output(i).asnumpy()[0])

        self.lock.release()

        return tvm_output
