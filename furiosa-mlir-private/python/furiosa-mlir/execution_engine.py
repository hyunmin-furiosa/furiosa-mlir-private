from ._mlir_libs import _furiosaMlirExecutionEngine as _furiosa_execution_engine

from furiosa_mlir.runtime.np_to_tensor import *

import ctypes

__all__ = [
    "ExecutionEngine",
]

class ExecutionEngine(_furiosa_execution_engine.ExecutionEngine):
    def invoke(self, name, inputs=[], outputs=[]):
        """Invoke a function with the list of numpy arguments.
        All arguments must be numpy array object.
        Raise a RuntimeError if the function isn't found.
        """
        num_arguments = len(inputs) + len(outputs)
        packed_args = (ctypes.c_void_p * num_arguments)()
        for argNum in range(len(inputs)):
            pointer = ctypes.pointer(get_ranked_tensor_descriptor(inputs[argNum]))
            packed_args[argNum] = ctypes.cast(pointer, ctypes.c_void_p)
        for argNum in range(len(outputs)):
            pointer = ctypes.pointer(get_ranked_tensor_descriptor(outputs[argNum]))
            packed_args[len(inputs) + argNum] = ctypes.cast(pointer, ctypes.c_void_p)
        packed_args_ptr = ctypes.cast(packed_args, ctypes.c_void_p).value
        self.raw_invoke(name, num_arguments, len(inputs), packed_args_ptr)
