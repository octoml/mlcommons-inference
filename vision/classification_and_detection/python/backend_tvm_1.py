"""
TVM backend for MLPerf inference vision benchmark

Developers: Alexander Peskov, Thierry Moreau, Grigori Fursin
"""
import backend

from tvm.contrib.async_launcher import AsyncGraphExecutor

import numpy as np
import os


class BackendTVM1(backend.Backend):
    def __init__(self):
        super(BackendTVM1, self).__init__()
        self.executor = None

    def version(self):
        return "N/A : TODO"

    def name(self):
        """Name of the runtime."""
        return "tvm"

    def image_format(self):
        """Requested image_format. Use a more popular layout NCHW"""
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""
        self.executor = AsyncGraphExecutor(model_path)

        self.inputs = inputs
        self.outputs = outputs

        if not inputs:
            self.inputs = [str(idx) for idx in range(self.executor.get_num_outputs())]
        if not outputs:
            self.outputs = [str(idx) for idx in range(self.executor.get_num_outputs())]

        executor_type = os.environ.get('MLPERF_TVM_EXECUTOR', 'graph')
        assert executor_type in ("graph", "debug")

        return self

    def predict(self, feed):
        """Run the prediction."""
        inputs = [None] * len(self.inputs)
        for i_name, i_data in feed.items():
            input_idx = self.inputs.index(i_name)
            inputs[input_idx] = i_data

        # Run TVM inference
        res = self.executor.infer(inputs)

        # Assume that only one output produced
        tvm_output = [res]

        return tvm_output

