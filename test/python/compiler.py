from furiosa_mlir import *
from furiosa_mlir.compiler import *
from furiosa_mlir.ir import *
from furiosa_mlir.passmanager import *

def test_compiler_high_level():
    with Context() as ctx, Location.unknown():
        module = Module.parse(
r"""
module {
  func.func @kernel(%arg0: tensor<64x64x64xi8>, %arg1: tensor<64x64x64xi8>) -> (tensor<64x64x64xi8>) attributes { target = #furiosa.target<npu 0 pe 0:0> } {
    %0 = tensor.empty() : tensor<64x64x64xi8>
    %a_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %b_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp : (tensor<64x64x64xi8>, tensor<64x64x64xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<64x64x64xi8>
    return %1 : tensor<64x64x64xi8>
  }
}
"""
        )

        compiler = Compiler()
        compiler.compile(module)

test_compiler_high_level()
