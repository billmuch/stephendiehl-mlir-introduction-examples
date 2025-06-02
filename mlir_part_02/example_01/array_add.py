import ctypes
import numpy as np
from np_memref import MemRefDescriptor, numpy_to_memref


def main():
    # Load the shared library
    lib = ctypes.CDLL("./libarray_add.so")

    # Get the C interface function
    array_add = lib._mlir_ciface_array_add
    array_add.argtypes = [
        ctypes.POINTER(MemRefDescriptor)
    ] * 3  # input1, input2, output
    array_add.restype = None

    # Create sample input arrays
    size = 1024
    a = np.ones(size, dtype=np.float32)
    b = np.ones(size, dtype=np.float32) * 2
    c = np.zeros(size, dtype=np.float32)

    # Convert arrays to MemRef descriptors
    a_desc = numpy_to_memref(a)
    b_desc = numpy_to_memref(b)
    c_desc = numpy_to_memref(c)

    # Call the function
    array_add(ctypes.byref(a_desc), ctypes.byref(b_desc), ctypes.byref(c_desc))

    # Verify results
    expected = a + b
    np.testing.assert_array_almost_equal(c, expected)
    print("Array addition successful!")
    print(f"First few elements: {c[:5]}")  # Should show [3.0, 3.0, 3.0, 3.0, 3.0]


if __name__ == "__main__":
    main()
