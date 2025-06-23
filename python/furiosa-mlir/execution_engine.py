from ._mlir_libs import _furiosaMlirExecutionEngine as _furiosa_execution_engine
import ctypes

__all__ = [
    "ExecutionEngine",
]

class ExecutionEngine(_furiosa_execution_engine.ExecutionEngine):
    def invoke(self, name, ctypes_inputs=[], ctypes_outputs=[]):
        """Invoke a function with the list of ctypes arguments.
        All arguments must be pointers.
        Raise a RuntimeError if the function isn't found.
        """
        num_arguments = len(ctypes_inputs) + len(ctypes_outputs)
        packed_args = (ctypes.c_void_p * num_arguments)()
        for argNum in range(len(ctypes_inputs)):
            packed_args[argNum] = ctypes.cast(ctypes_inputs[argNum], ctypes.c_void_p)
        for argNum in range(len(ctypes_outputs)):
            packed_args[len(ctypes_inputs) + argNum] = ctypes.cast(
                ctypes_outputs[argNum], ctypes.c_void_p
            )
        packed_args_ptr = ctypes.cast(packed_args, ctypes.c_void_p).value
        self.raw_invoke(name, num_arguments, len(ctypes_inputs), packed_args_ptr)
