import ctypes
import numpy as np
from ctypes import c_void_p, c_longlong, Structure


class MemRef2DDescriptor(Structure):
    """Structure matching MLIR's 2D MemRef descriptor"""

    _fields_ = [
        ("allocated", c_void_p),  # Allocated pointer
        ("aligned", c_void_p),  # Aligned pointer
        ("offset", c_longlong),  # Offset in elements
        ("shape", c_longlong * 2),  # Array shape (2D)
        ("stride", c_longlong * 2),  # Strides in elements
    ]

def numpy_to_memref2d(arr):
    """Convert a 2D NumPy array to a MemRef descriptor"""
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    desc = MemRef2DDescriptor()
    desc.allocated = arr.ctypes.data_as(c_void_p)
    desc.aligned = desc.allocated
    desc.offset = 0
    desc.shape[0] = arr.shape[0]
    desc.shape[1] = arr.shape[1]
    desc.stride[0] = arr.strides[0] // arr.itemsize
    desc.stride[1] = arr.strides[1] // arr.itemsize

    return desc


def run_conv2d():
    """Run 2D convolution using MLIR compiled module"""
    # Create input arrays
    input_matrix = np.ones((10, 10), dtype=np.float32)
    conv_filter = np.arange(9, dtype=np.float32).reshape(3, 3)
    result = np.zeros((8, 8), dtype=np.float32)

    # Load compiled module
    module = ctypes.CDLL("./libconv2d.dylib")

    # Prepare MemRef descriptors
    input_memref = numpy_to_memref2d(input_matrix)
    filter_memref = numpy_to_memref2d(conv_filter)
    result_memref = numpy_to_memref2d(result)

    conv_2d = module._mlir_ciface_conv_2d
    # Set function argument types
    conv_2d.argtypes =  [
        ctypes.POINTER(MemRef2DDescriptor)
    ] * 3

    # Call the function
    conv_2d(
        ctypes.byref(input_memref),
        ctypes.byref(filter_memref),
        ctypes.byref(result_memref)
    )

    return result


if __name__ == "__main__":
    result = run_conv2d()
    print(result)
