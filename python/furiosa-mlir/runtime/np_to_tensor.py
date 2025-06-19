import numpy as np
import ctypes

try:
    import ml_dtypes
except ModuleNotFoundError:
    # The third-party ml_dtypes provides some optional low precision data-types for NumPy.
    ml_dtypes = None


class C128(ctypes.Structure):
    """A ctype representation for MLIR's Double Complex."""

    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


class C64(ctypes.Structure):
    """A ctype representation for MLIR's Float Complex."""

    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class F16(ctypes.Structure):
    """A ctype representation for MLIR's Float16."""

    _fields_ = [("f16", ctypes.c_int16)]


class BF16(ctypes.Structure):
    """A ctype representation for MLIR's BFloat16."""

    _fields_ = [("bf16", ctypes.c_int16)]

class F8E5M2(ctypes.Structure):
    """A ctype representation for MLIR's Float8E5M2."""

    _fields_ = [("f8E5M2", ctypes.c_int8)]


# https://stackoverflow.com/questions/26921836/correct-way-to-test-for-numpy-dtype
def as_ctype(dtp):
    """Converts dtype to ctype."""
    if dtp == np.dtype(np.complex128):
        return C128
    if dtp == np.dtype(np.complex64):
        return C64
    if dtp == np.dtype(np.float16):
        return F16
    if ml_dtypes is not None and dtp == ml_dtypes.bfloat16:
        return BF16
    if ml_dtypes is not None and dtp == ml_dtypes.float8_e5m2:
        return F8E5M2
    return np.ctypeslib.as_ctypes_type(dtp)


def to_numpy(array):
    """Converts ctypes array back to numpy dtype array."""
    if array.dtype == C128:
        return array.view("complex128")
    if array.dtype == C64:
        return array.view("complex64")
    if array.dtype == F16:
        return array.view("float16")
    assert not (
        array.dtype == BF16 and ml_dtypes is None
    ), f"bfloat16 requires the ml_dtypes package, please run:\n\npip install ml_dtypes\n"
    if array.dtype == BF16:
        return array.view("bfloat16")
    assert not (
        array.dtype == F8E5M2 and ml_dtypes is None
    ), f"float8_e5m2 requires the ml_dtypes package, please run:\n\npip install ml_dtypes\n"
    if array.dtype == F8E5M2:
        return array.view("float8_e5m2")
    return array

def make_nd_tensor_descriptor(rank, dtype):
    class TensorDescriptor(ctypes.Structure):
        """Builds an empty descriptor for the given rank/dtype, where rank>0."""

        _fields_ = [
            ("allocated", ctypes.c_longlong),
            ("aligned", ctypes.POINTER(dtype)),
            ("offset", ctypes.c_longlong),
            ("shape", ctypes.c_longlong * rank),
        ]

    return TensorDescriptor

def get_ranked_tensor_descriptor(nparray):
    ctp = as_ctype(nparray.dtype)

    x = make_nd_tensor_descriptor(nparray.ndim, ctp)()
    x.allocated = nparray.ctypes.data
    x.aligned = nparray.ctypes.data_as(ctypes.POINTER(ctp))
    x.offset = ctypes.c_longlong(0)
    x.shape = nparray.ctypes.shape

    return x
