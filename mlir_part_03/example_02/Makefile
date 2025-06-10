# Default target
all: omp

# Optimize MLIR with proper dialect conversion
omp_opt.mlir: omp.mlir
	mlir-opt omp.mlir \
	  --convert-openmp-to-llvm \
	  --convert-scf-to-cf \
	  --convert-cf-to-llvm \
	  --convert-index-to-llvm \
	  --convert-arith-to-llvm \
	  --convert-func-to-llvm \
	  --finalize-memref-to-llvm \
	  --reconcile-unrealized-casts \
	  -o omp_opt.mlir

# Translate to LLVM IR
omp.ll: omp_opt.mlir
	mlir-translate omp_opt.mlir -mlir-to-llvmir -o omp.ll

# Compile to object file
omp.o: omp.ll
	llc -filetype=obj --relocation-model=pic omp.ll -o omp.o

# Link to executable
omp: omp.o
	clang -lomp omp.o -o omp

# Clean generated files
clean:
	rm -f omp_opt.mlir omp.ll omp.o omp

.PHONY: all clean



