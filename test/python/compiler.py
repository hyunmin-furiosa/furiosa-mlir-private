from furiosa_mlir import *
from furiosa_mlir.compiler import *
from furiosa_mlir.execution_engine import *
from furiosa_mlir.ir import *
from furiosa_mlir.passmanager import *

import furiosa_mlir.dialects.furiosa as furiosa

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

def test_compiler_and_execution_engine():
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

        arr0 = np.random.randint(0, 2, size=(64, 64, 64), dtype=np.int8)
        arr1 = np.random.randint(0, 2, size=(64, 64, 64), dtype=np.int8)
        arr2 = np.zeros((64, 64, 64), dtype=np.int8)

        target = furiosa.TargetAttr.get(npu=0, pe_begin=0, pe_end=0)
        execution_engine = ExecutionEngine(module, target)
        execution_engine.invoke("kernel", [arr0, arr1], [arr2])

        expected = np.einsum("nij,njk->nik", arr0, arr1)
        print(np.array_equal(arr2, expected))

test_compiler_high_level()
test_compiler_and_execution_engine()
