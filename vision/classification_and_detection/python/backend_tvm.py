"""
TBD
"""
import backend

import tvm
from tvm.contrib import graph_executor

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
        return "N/A : TODO"

    def name(self):
        """Name of the runtime."""
        return "tvm"

    def image_format(self):
        """Requested image_format. Use a more popular layout NCHW"""
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
        # First attempt to detect input and output names via ONNX run time.
        # See backend_onnxruntime.py
        #
        # Even if inputs/outputs can be defined by MLPerf
        # TVM will need extra info about shapes to be properly initialized!

        self.inputs = inputs
        self.outputs = outputs

        # TODO(@peskov): who specify that?? Only outside? Looks like TVM specific WA
        # Max batch size should be passed from main function
        self.max_batchsize = max_batchsize

        ctx = tvm.cpu(0)
        self.lib = tvm.runtime.load_module(model_path)
        self.graph = graph_executor.GraphModule(self.lib['default'](ctx))

        # TODO(@apekov): Check if provided inputs/outputs match with presented in model
        # TODO(@apekov): Is there function to get names of inputs/outputs? meanwhile fill it with fake names
        if not inputs:
            self.inputs = [str(idx) for idx in range(self.graph.get_num_outputs())]
        if not outputs:
            self.outputs = [str(idx) for idx in range(self.graph.get_num_outputs())]

        self.executor_type = os.environ.get('MLPERF_TVM_EXECUTOR', 'graph')

        if self.executor_type in ("graph", "debug"):
            pass
        elif self.executor_type in ("vm", "vm-stateful"):
            raise Exception("VM mode is UNSUPPORTED ...")

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
