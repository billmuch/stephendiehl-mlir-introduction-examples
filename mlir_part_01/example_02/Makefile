all: libsimple.so

simple_opt.mlir: simple.mlir
	mlir-opt $< \
	  --convert-func-to-llvm \
	  --convert-math-to-llvm \
	  --convert-index-to-llvm \
	  --convert-scf-to-cf \
	  --convert-cf-to-llvm \
	  --convert-arith-to-llvm \
	  --reconcile-unrealized-casts \
	  -o $@

simple.ll: simple_opt.mlir
	mlir-translate $< -mlir-to-llvmir -o $@

simple.o: simple.ll
	llc -filetype=obj --relocation-model=pic $< -o $@

libsimple.so: simple.o
	clang -shared -fPIC $< -o $@

clean:
	rm -f simple_opt.mlir simple.ll simple.o libsimple.so

.PHONY: all clean
