// RUN: furiosa-mlir-opt --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named))'

module {
  func.func @kernel(%arg0: tensor<1x5x3xf32>, %arg1: tensor<1x3x6xf32>) -> (tensor<1x5x6xf32>) attributes { target = #furiosa.target<npu 0 pe 0:0> } {
    %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp : (tensor<1x5x3xf32>, tensor<1x3x6xf32>, tensor<1xf32>, tensor<1xf32>)  -> tensor<1x5x6xf32>
    return %0 : tensor<1x5x6xf32>
  }
  func.func @main() {
    %a = tensor.empty() : tensor<1x5x3xf32>
    %b = tensor.empty() : tensor<1x3x6xf32>
    %c = func.call @kernel(%a, %b) { target = #furiosa.target<npu 0 pe 0:0> } : (tensor<1x5x3xf32>, tensor<1x3x6xf32>) -> (tensor<1x5x6xf32>)
    return
  }
}
