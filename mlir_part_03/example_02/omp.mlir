memref.global constant @input : memref<10xf32> = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]>

llvm.func @printf(!llvm.ptr, ...) -> i32
llvm.mlir.global private constant @fmt("%f\0A\00") {addr_space = 0 : i32}

func.func private @kernel(%input: memref<10xf32>, %output: memref<10xf32>) {
  %loop_ub = llvm.mlir.constant(9 : i32) : i32
  %loop_lb = llvm.mlir.constant(0 : i32) : i32
  %loop_step = llvm.mlir.constant(1 : i32) : i32

  omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : i32 = (%loop_lb) to (%loop_ub) inclusive step (%loop_step) {
          %ix = arith.index_cast %i : i32 to index
          %input_val = memref.load %input[%ix] : memref<10xf32>
          %two = arith.constant 2.0 : f32
          %result = arith.mulf %input_val, %two : f32
          memref.store %result, %output[%ix] : memref<10xf32>
          omp.yield
        }
      }
    omp.barrier
    omp.terminator
  }

  return
}

func.func private @main() {
  %input = memref.get_global @input : memref<10xf32>
  %output = memref.alloc() : memref<10xf32>

  call @kernel(%input, %output) : (memref<10xf32>, memref<10xf32>) -> ()

  %lb = index.constant 0
  %ub = index.constant 10 
  %step = index.constant 1

  %fs = llvm.mlir.addressof @fmt : !llvm.ptr

  scf.for %iv = %lb to %ub step %step {
    %el = memref.load %output[%iv] : memref<10xf32>
    %el_double = arith.extf %el : f32 to f64
    llvm.call @printf(%fs, %el_double) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
  }

  return
}


