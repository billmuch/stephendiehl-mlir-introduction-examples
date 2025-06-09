module {
  func.func @conv_2d(%input: memref<10x10xf32>, %filter: memref<3x3xf32>, %output: memref<8x8xf32>) 
        attributes {llvm.emit_c_interface} {
    // Loop over the output matrix dimensions (8x8)
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        // Use affine.parallel to accumulate values into %acc using iter_args
        %zero = arith.constant 0.0 : f32
        %acc = affine.for %fi = 0 to 3 iter_args(%acc = %zero) -> (f32) {
          %acc_inner = affine.for %fj = 0 to 3 iter_args(%acc_inner = %acc) -> (f32) {
            // Load filter value
            %filter_val = affine.load %filter[%fi, %fj] : memref<3x3xf32>

            // Load corresponding input value from the input matrix
            %input_val = affine.load %input[%i + %fi, %j + %fj] : memref<10x10xf32>

            // Multiply input value with filter value
            %prod = arith.mulf %input_val, %filter_val : f32

            // Add product to the accumulator
            %new_acc = arith.addf %acc_inner, %prod : f32
            affine.yield %new_acc : f32
          }
          affine.yield %acc_inner : f32
        }

        // Store the accumulated result in the output matrix
        affine.store %acc, %output[%i, %j] : memref<8x8xf32>
      }
    }
    return
  }
}

