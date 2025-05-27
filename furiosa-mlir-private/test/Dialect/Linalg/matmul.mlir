// RUN: furiosa-mlir-opt -linalg-generalize-to-contract-ops

module {
  func.func @kernel(%arg0: tensor<64x64x64xf32>, %arg1: tensor<64x64x64xf32>) -> tensor<64x64x64xf32> attributes {target = #furiosa.target<npu 0 pe 0 : 0>} {
    %0 = tensor.empty() : tensor<64x64x64xf32>
    %1 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %3 = tensor.empty() : tensor<64x64x64xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<64x64x64xf32>) -> tensor<64x64x64xf32>
    %5 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x64x64xf32>, tensor<64x64x64xf32>) outs(%4 : tensor<64x64x64xf32>) -> tensor<64x64x64xf32>
    return %5 : tensor<64x64x64xf32>
  }
  func.func @main() {
    %0 = tensor.empty() : tensor<64x64x64xf32>
    %1 = tensor.empty() : tensor<64x64x64xf32>
    %2 = call @kernel(%0, %1) {target = #furiosa.target<npu 0 pe 0 : 0>} : (tensor<64x64x64xf32>, tensor<64x64x64xf32>) -> tensor<64x64x64xf32>
    return
  }
}
