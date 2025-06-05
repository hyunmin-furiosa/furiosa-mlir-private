// RUN: furiosa-mlir-opt --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named))'

module {
  func.func @kernel(%arg0: tensor<64x64x64xf8E5M2>, %arg1: tensor<64x64x64xf8E5M2>) -> (tensor<64x64x64xf8E5M2>) attributes { target = #furiosa.target<npu 0 pe 0:0> } {
    %0 = tensor.empty() : tensor<64x64x64xf8E5M2>
    %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
    %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
    %1 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp : (tensor<64x64x64xf8E5M2>, tensor<64x64x64xf8E5M2>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<64x64x64xf8E5M2>
    return %1 : tensor<64x64x64xf8E5M2>
  }
  func.func @main() {
    %0 = tensor.empty() : tensor<64x64x64xf8E5M2>
    %1 = tensor.empty() : tensor<64x64x64xf8E5M2>
    %3 = call @kernel(%0, %1) { target = #furiosa.target<npu 0 pe 0:0> } : (tensor<64x64x64xf8E5M2>, tensor<64x64x64xf8E5M2>) -> tensor<64x64x64xf8E5M2>
    return
  }
}
