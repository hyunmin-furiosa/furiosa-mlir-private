from ._mlir_libs import _furiosaMlirExecutionEngine as _furiosa_execution_engine
import ctypes

__all__ = [
    "ExecutionEngine",
]

class ExecutionEngine(_furiosa_execution_engine.ExecutionEngine):
    def invoke(self, name, *ctypes_args):
        """Invoke a function with the list of ctypes arguments.
        All arguments must be pointers.
        Raise a RuntimeError if the function isn't found.
        """
        # func = self.lookup(name)
        # packed_args = (ctypes.c_void_p * len(ctypes_args))()
        # for argNum in range(len(ctypes_args)):
        #     packed_args[argNum] = ctypes.cast(ctypes_args[argNum], ctypes.c_void_p)
        # func(packed_args)
