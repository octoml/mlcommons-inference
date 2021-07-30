"""
TVM backend with PyTorch models (https://github.com/apache/tvm)

TBD: for now hardwired for resnet50_INT8bit_quantized.pt - need to make it more universal

Developer(s): grigori@octoml.ai
              Alexader Peskov
"""

import backend

import torch

import tvm
from tvm import runtime
from tvm.contrib import graph_executor, graph_runtime
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.driver.tvmc.common import convert_graph_layout
from tvm.relay.op.contrib.dnnl import partition_for_dnnl

import numpy as np

import re
import os

import multiprocessing

g_graph = None

class BackendTVM(backend.Backend):
    def __init__(self):
        super(BackendTVM, self).__init__()
        self.arena_num = 1
        self.arena_size = multiprocessing.cpu_count()
        self.lib = None
        self.graph = None
        self.executor_type = None
        self.max_batchsize = None
        self.pool = None

    def version(self):
        return tvm.__version__

    def name(self):
        """Name of the runtime."""
        return "tvm-pytorch"

    def image_format(self):
        """image_format."""
        # For now force NCHW.
        return "NCHW"

    def create_omp_args(self, arena_idx):
        idx_start = self.arena_size * arena_idx
        cur_arena_size = min(multiprocessing.cpu_count() - idx_start, self.arena_size)
        # idx_end = idx_start + cur_arena_size

        # OMP_PLACES="{N},{N+1},{N+2},...,{N+SZ}"
        # arena_places_str = "{" + "},{".join(str(i) for i in range(idx_start, idx_end)) + "}"

        return {
                "TVM_NUM_THREADS": str(cur_arena_size),
                "OMP_NUM_THREADS": str(cur_arena_size),
                # "OMP_PLACES": arena_places_str,
                # "OMP_PROC_BIND": "true"
        }

    @staticmethod
    def set_omp_envs(omp_args):
        for env_arg in omp_args:
            os.environ[env_arg[0]] = env_arg[1]

    def load_impl(self, model_path, inputs, outputs, max_batchsize):
        """Load model and find input/outputs from the model file."""

        # First attempt to detect input and output names via ONNX run time.
        # See backend_onnxruntime.py
        #
        # Even if inputs/outputs can be defined by MLPerf
        # TVM will need extra info about shapes to be properly initialized!

        compiled_model='/tmp/compiled-model-tvm-pytorch.so'

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
        self.max_batchsize = max_batchsize

        ctx = tvm.cpu(0)
        self.tvm_ctx = ctx

        executor=os.environ.get('MLPERF_TVM_EXECUTOR','graph')
        self.tvm_executor=executor

        if not os.path.isfile(compiled_model):
           print ('')
           print ('TVM PyTorch: load model ...')
           print ('')

           pytorch_model = torch.jit.load(model_path)
           pytorch_model.eval()

           shape_list = [("input", [max_batchsize, 3, 224, 224])]

           mod, params = relay.frontend.from_pytorch(pytorch_model, shape_list)

           mod["main"] = bind_params_by_name(mod["main"], params)


           mod = transform.FoldConstant()(mod)

           #  move to NHWC layout, prerequisite for DNNL partitioning
           mod = convert_graph_layout(mod, "NHWC")
           mod = transform.FoldConstant()(mod)

           # partitioning for DNNL
           mod = partition_for_dnnl(mod)

           #######################################################################
           # Init TVM
           # TBD: add tvm platform selector
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

           self.lib = None

           # Needed for prediction
           if executor == "graph" or executor == "debug":
               from tvm.contrib import graph_executor

               self.lib=relay.build(mod, target=tvm_target)

               print ('')
               print ('TVM: init graph engine ...')
               print ('')

               self.graph = graph_executor.GraphModule(self.lib['default'](ctx))
#               self.sess = graph_executor.GraphModule(self.lib['default'](ctx))

               print ('')
               print ('TVM: exporting model as library ...')
               print ('')

               self.lib.export_library(compiled_model)

           # For now hardwire inputs/outputs for just 1 model
           # TBD: take info from the CK package for a given model!

           print ('')
           print ('TVM: model ready ...')
           print ('')

        else:
           print ('TVM: pre-loading model {} ...'.format(compiled_model))

           if executor == "graph" or executor == "debug":
               from tvm.contrib import graph_executor
               from tvm import runtime

               self.lib = runtime.load_module(compiled_model)

               self.graph = graph_executor.GraphModule(self.lib['default'](ctx))
#               self.sess = graph_executor.GraphModule(self.lib['default'](ctx))

#               with open('/tmp/deploy.json', "r") as f:
#                  json_data = f.read()

#               with open('/tmp/deploy.params', "rb") as f:
#                  parameters_data = f.read()

#               self.lib = runtime.load_module('/tmp/deploy.so')

#               self.graph = graph_runtime.create(json_data, self.lib, runtime.cpu())
#               self.graph.load_params(parameters_data)



        self.inputs=['input']
        self.outputs=['output']

        return self





    def predict_old(self, feed):

        sess = self.sess

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





    def predict_impl(self, feed):
        if self.executor_type in ("vm", "vm-stateful"):
            raise Exception("VM mode is UNSUPPORTED ...")
        else:
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

                input_idx = self.inputs.index(iname)
                self.graph.set_input(input_idx, tvm.nd.array(data))

            # Run TVM inference
            self.graph.run()

            # Process TVM outputs
            tvm_output = []
            for i in range(self.graph.get_num_outputs()):
                # Take only the output of batch size for dynamic batches
                tvm_output.append(self.graph.get_output(i).asnumpy()[:batch_size])

        return tvm_output



    @staticmethod
    def _worker_initializer(model_path, inputs, outputs, max_batchsize, omp_envs):
        BackendTVM.set_omp_envs(omp_envs)
        global g_graph
        g_graph = BackendTVM()
        g_graph.arena_num = 1
        g_graph.load_impl(model_path, inputs, outputs, max_batchsize)

    @staticmethod
    def _worker_handler(feed):
        global g_graph
        return g_graph.predict(feed)

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""
        self.load_impl(model_path, inputs, outputs, self.max_batchsize)

        if self.arena_num > 1:
            self.pool = multiprocessing.Pool(self.arena_num,
                                             initializer=self._worker_initializer,
                                             initargs=(model_path, inputs, outputs, self.max_batchsize,
                                                       self.create_omp_args(0))
                                             )

        # TODO(@apeskov): do we really have to return self ??
        return self

    def predict(self, feed):
        """Run the prediction."""
        if self.arena_num > 1:
            resp = self.pool.apply_async(self._worker_handler, args=(feed,))
            return resp.get()
        else:
            return self.predict_impl(feed)
