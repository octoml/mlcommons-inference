"""
TVM backend with ONNX models (https://github.com/apache/tvm)

Developer(s): Grigori Fursin, grigori@octoml.ai
              Alexander Peskov
"""

import onnx
import onnxruntime as rt

import backend

import tvm
from tvm import runtime
from tvm.contrib import graph_executor, graph_runtime
from tvm import relay

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
        return rt.__version__

    def name(self):
        """Name of the runtime."""
        return "tvm-onnx"

    def image_format(self):
        """image_format."""
        # We use ONNX format, which is always NCHW.
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

        compiled_model='/tmp/compiled-model-tvm-onnx.so'

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

        self.max_batchsize = max_batchsize

        ctx = tvm.cpu(0)
        self.tvm_ctx = ctx

        executor=os.environ.get('MLPERF_TVM_EXECUTOR','graph')
        self.tvm_executor=executor

        if not os.path.isfile(compiled_model):
           print ('')
           print ('TVM ONNX: initialize runtime to get some model parameters ...')
           print ('')

           opt = rt.SessionOptions()

           tmp_sess = rt.InferenceSession(model_path, opt)

           if not inputs:
               self.inputs = [meta.name for meta in tmp_sess.get_inputs()]
           if not outputs:
               self.outputs = [meta.name for meta in tmp_sess.get_outputs()]

           # Detect shapes and set max batch size.
           # If batch size is < max batch size, fill in with empty ones
           # In the future, we should support dynamic batch sizes in TVM

           shape_dict = {}
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
               input_shape[0] = max_batchsize
               shape_dict[input_name] = tuple(input_shape)

           print ('')
           print ('TVM: input shape(s): '+str(shape_dict))
           print ('TVM: input type: '+str(dtype_dict))
           print ('TVM: outputs: '+str(self.outputs))
           print ('')

   #        input('xyz')

           self.input_shapes = shape_dict

           # We do not need ONNX runtime anymore
           del tmp_sess

           # Load model via ONNX to be used with TVM
           print ('')
           print ('TVM ONNX: load model ...')
           onnx_model = onnx.load(model_path)

           print ('')
           print ('TVM ONNX: import model ...')
           print ('')
           # Extra param: opset=12
           mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

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

           # Model updates
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


           # Needed for prediction

           self.lib = None
           if executor == "graph" or executor == "debug":
               from tvm.contrib import graph_executor

               # Without history
               with tvm.transform.PassContext(opt_level=opt_lvl, config=build_conf):
                   graph_module = relay.build(mod,
                                              target=tvm_target,
                                              params=params)
               self.lib = graph_module

               print ('')
               print ('TVM: init graph engine ...')
               print ('')

               self.sess = graph_executor.GraphModule(self.lib['default'](ctx))
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

           if self.lib!=None:
              print ('')
              print ('TVM: exporting model as library ...')
              print ('')

              self.lib.export_library(compiled_model)

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
               self.sess = graph_executor.GraphModule(self.lib['default'](ctx))

               # Temporally hardwire inputs/outputs (need to load from a model)
               self.inputs = ['input_tensor:0']
               self.outputs = ['ArgMax:0']

#               with open('/tmp/deploy.json', "r") as f:
#                  json_data = f.read()
#
#               with open('/tmp/deploy.params', "rb") as f:
#                  parameters_data = f.read()
#
#               self.lib = runtime.load_module('/tmp/deploy.so')
#
#               self.graph = graph_runtime.create(json_data, self.lib, runtime.cpu())
#               self.graph.load_params(parameters_data)





           else:
               raise Exception("VM mode is UNSUPPORTED with preloaded model...")

        return self


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


    def predict_old(self, feed):
        """Run the prediction."""

        executor = self.tvm_executor

        sess = self.sess

#        self.lock.acquire()

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
            tvm_output = []
            for i in range(sess.get_num_outputs()):
                # Take only the output of batch size for dynamic batches
                tvm_output.append(sess.get_output(i).asnumpy()[:batch_size])

#        self.lock.release()

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
