all: libarray_add.so

array_add_opt.mlir: array_add.mlir
	mlir-opt $< \
	  --convert-tensor-to-linalg \
	  --convert-linalg-to-loops \
	  --convert-scf-to-cf \
	  --convert-cf-to-llvm \
	  --convert-math-to-llvm \
	  --convert-arith-to-llvm \
	  --convert-func-to-llvm \
	  --convert-index-to-llvm \
	  --finalize-memref-to-llvm \
	  --reconcile-unrealized-casts \
	  -o $@

array_add_opt.ll: array_add_opt.mlir
	mlir-translate $< \
	  -mlir-to-llvmir \
	  -o $@

array_add.o: array_add_opt.ll
	llc -filetype=obj --relocation-model=pic $< -o $@

libarray_add.so: array_add.o
	clang -shared $< -o $@

clean:
	rm -f array_add_opt.mlir array_add_opt.ll array_add.o libarray_add.so

.PHONY: all clean
