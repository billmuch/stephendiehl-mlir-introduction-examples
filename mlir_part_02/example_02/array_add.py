import os
import subprocess
import numpy as np
from ctypes import CFUNCTYPE, POINTER
from np_memref import MemRefDescriptor, numpy_to_memref

import llvmlite

# Disable typed pointers for LLVM 15+ compatibility
llvmlite.opaque_pointers_enabled = True

import llvmlite.binding as llvm  # noqa: E402

# Initialize LLVM
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()


def compile_mlir_to_llvm(mlir_file_path):
    """Compile MLIR code to LLVM IR using mlir-opt and mlir-translate"""
    # Run mlir-opt to lower to LLVM dialect
    opt_cmd = [
        "mlir-opt",
        mlir_file_path,
        "--convert-tensor-to-linalg",
        "--convert-linalg-to-loops",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-math-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-func-to-llvm",
        "--convert-index-to-llvm",
        "--finalize-memref-to-llvm",
        "--reconcile-unrealized-casts",
    ]

    try:
        opt_result = subprocess.run(opt_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error running mlir-opt:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    # Run mlir-translate to convert to LLVM IR
    translate_cmd = ["mlir-translate", "--mlir-to-llvmir"]
    try:
        translate_result = subprocess.run(
            translate_cmd,
            input=opt_result.stdout,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("Error running mlir-translate:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    return translate_result.stdout


def create_execution_engine():
    """Create an ExecutionEngine suitable for JIT"""
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def compile_and_load_mlir(mlir_file_path):
    """Compile MLIR code and load it into a JIT engine"""
    # Convert MLIR to LLVM IR
    llvm_ir = compile_mlir_to_llvm(mlir_file_path)

    # Create module from LLVM IR
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()

    # Create execution engine and add module
    engine = create_execution_engine()
    engine.add_module(mod)
    engine.finalize_object()

    return engine, mod


def main():
    # Get path to MLIR file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_file = os.path.join(current_dir, "array_add.mlir")

    # Compile and load the MLIR code
    engine, mod = compile_and_load_mlir(mlir_file)

    # Get function pointer to the compiled function
    func_ptr = engine.get_function_address("_mlir_ciface_array_add")

    # Create the ctypes function wrapper
    array_add = CFUNCTYPE(
        None,
        POINTER(MemRefDescriptor),
        POINTER(MemRefDescriptor),
        POINTER(MemRefDescriptor),
    )(func_ptr)

    # Create test arrays
    size = 1024
    a = np.ones(size, dtype=np.float32)
    b = np.ones(size, dtype=np.float32) * 2
    c = np.zeros(size, dtype=np.float32)

    # Convert arrays to MemRef descriptors
    a_desc = numpy_to_memref(a)
    b_desc = numpy_to_memref(b)
    c_desc = numpy_to_memref(c)

    # Call the JIT-compiled function
    array_add(a_desc, b_desc, c_desc)

    # Verify results
    expected = a + b
    np.testing.assert_array_almost_equal(c, expected)
    print("Array addition successful!")
    print(f"First few elements: {c[:5]}")  # Should show [3.0, 3.0, 3.0, 3.0, 3.0]


if __name__ == "__main__":
    main()
