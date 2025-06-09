# Makefile for MLIR conv2d example

# Default target
all: libconv2d.dylib

# Optimize MLIR
conv_2d_opt.mlir: conv_2d.mlir
	mlir-opt conv_2d.mlir \
	  --affine-loop-normalize \
	  --affine-parallelize \
	  -lower-affine \
	  --convert-scf-to-cf \
	  --convert-cf-to-llvm \
	  --convert-math-to-llvm \
	  --convert-arith-to-llvm \
	  --finalize-memref-to-llvm \
	  --convert-func-to-llvm \
	  --reconcile-unrealized-casts \
	  -o conv_2d_opt.mlir

# Translate to LLVM IR
conv_2d.ll: conv_2d_opt.mlir
	mlir-translate conv_2d_opt.mlir -mlir-to-llvmir -o conv_2d.ll

# Compile to object file
conv_2d.o: conv_2d.ll
	llc -filetype=obj --relocation-model=pic conv_2d.ll -o conv_2d.o

# Compile to shared library
libconv2d.dylib: conv_2d.o
	clang -shared -lomp -o libconv2d.dylib conv_2d.o

# Clean up generated files
clean:
	rm -f conv_2d_opt.mlir conv_2d.ll conv_2d.o libconv2d.dylib

.PHONY: all clean
