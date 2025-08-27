from furiosa_mlir import *
from furiosa_mlir.compiler import *
from furiosa_mlir.execution_engine import *
from furiosa_mlir.ir import *
from furiosa_mlir.passmanager import *

import furiosa_mlir.dialects.func as func
import furiosa_mlir.dialects.furiosa as furiosa
import furiosa_mlir.dialects.tosa as tosa

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

def test_minimum_dimension():
    n = 64
    i = 64
    j = 32
    k = 32
    with Context() as ctx, Location.unknown():
        i8_type = IntegerType.get_signless(8)
        arr0_type = RankedTensorType.get((n, i, j), i8_type)
        arr1_type = RankedTensorType.get((n, j, k), i8_type)
        arr2_type = RankedTensorType.get((n, i, k), i8_type)
        zp_type = RankedTensorType.get((1,), i8_type)
        zp_element = IntegerAttr.get(i8_type, 0)
        zp_attr = DenseElementsAttr.get([zp_element], zp_type)

        module = Module.create()
        function = func.FuncOp("kernel", ([arr0_type, arr1_type], [arr2_type]))
        function.attributes["target"] = furiosa.TargetAttr.get(npu=0, pe_begin=0, pe_end=0)
        function.add_entry_block()
        v0 = tosa.ConstOp(zp_attr)
        v1 = tosa.MatMulOp(arr2_type, function.arguments[0], function.arguments[1], v0, v0)
        function.entry_block.append(v0)
        function.entry_block.append(v1)
        function.entry_block.append(func.ReturnOp(v1))
        module.body.append(function)

        compiler = Compiler()
        compiler.compile(module)

        arr0 = np.random.randint(0, 2, size=(n, i, j), dtype=np.int8)
        arr1 = np.random.randint(0, 2, size=(n, j, k), dtype=np.int8)
        arr2 = np.zeros((n, i, k), dtype=np.int8)

        target = furiosa.TargetAttr.get(npu=0, pe_begin=0, pe_end=0)
        execution_engine = ExecutionEngine(module, target)
        execution_engine.invoke("kernel", [arr0, arr1], [arr2])

        expected = np.einsum("nij,njk->nik", arr0, arr1)
        print(np.array_equal(arr2, expected))

def test_various_dimensions():
    dim_list = []
    for n in range(64, 256, 64):
        for i in range(64, 256, 64):
            for j in range(32, 128, 32):
                for k in range(32, 128, 32):
                    dim_list.append([n, i, j, k])
    fail_list = []
    for [n, i, j, k] in dim_list:
        print(n, i, j, k)
        with Context() as ctx, Location.unknown():
            i8_type = IntegerType.get_signless(8)
            arr0_type = RankedTensorType.get((n, i, j), i8_type)
            arr1_type = RankedTensorType.get((n, j, k), i8_type)
            arr2_type = RankedTensorType.get((n, i, k), i8_type)
            zp_type = RankedTensorType.get((1,), i8_type)
            zp_element = IntegerAttr.get(i8_type, 0)
            zp_attr = DenseElementsAttr.get([zp_element], zp_type)

            module = Module.create()
            function = func.FuncOp("kernel", ([arr0_type, arr1_type], [arr2_type]))
            function.attributes["target"] = furiosa.TargetAttr.get(npu=0, pe_begin=0, pe_end=0)
            function.add_entry_block()
            v0 = tosa.ConstOp(zp_attr)
            v1 = tosa.MatMulOp(arr2_type, function.arguments[0], function.arguments[1], v0, v0)
            function.entry_block.append(v0)
            function.entry_block.append(v1)
            function.entry_block.append(func.ReturnOp(v1))
            module.body.append(function)

            compiler = Compiler()
            compiler.compile(module)

            arr0 = np.random.randint(0, 2, size=(n, i, j), dtype=np.int8)
            arr1 = np.random.randint(0, 2, size=(n, j, k), dtype=np.int8)
            arr2 = np.zeros((n, i, k), dtype=np.int8)

            target = furiosa.TargetAttr.get(npu=0, pe_begin=0, pe_end=0)
            execution_engine = ExecutionEngine(module, target)
            execution_engine.invoke("kernel", [arr0, arr1], [arr2])

            expected = np.einsum("nij,njk->nik", arr0, arr1)
            equal = np.array_equal(arr2, expected)
            print(equal)
            if not equal:
                fail_list.append((n, i, j, k, np.array_equal(arr2, expected)))
    print(fail_list)

def test_one_dimension():
    n = 192
    i = 192
    j = 64
    k = 64
    with Context() as ctx, Location.unknown():
        i8_type = IntegerType.get_signless(8)
        arr0_type = RankedTensorType.get((n, i, j), i8_type)
        arr1_type = RankedTensorType.get((n, j, k), i8_type)
        arr2_type = RankedTensorType.get((n, i, k), i8_type)
        zp_type = RankedTensorType.get((1,), i8_type)
        zp_element = IntegerAttr.get(i8_type, 0)
        zp_attr = DenseElementsAttr.get([zp_element], zp_type)

        module = Module.create()
        function = func.FuncOp("kernel", ([arr0_type, arr1_type], [arr2_type]))
        function.attributes["target"] = furiosa.TargetAttr.get(npu=0, pe_begin=0, pe_end=0)
        function.add_entry_block()
        v0 = tosa.ConstOp(zp_attr)
        v1 = tosa.MatMulOp(arr2_type, function.arguments[0], function.arguments[1], v0, v0)
        function.entry_block.append(v0)
        function.entry_block.append(v1)
        function.entry_block.append(func.ReturnOp(v1))
        module.body.append(function)

        compiler = Compiler()
        compiler.compile(module)

        arr0 = np.random.randint(0, 2, size=(n, i, j), dtype=np.int8)
        arr1 = np.random.randint(0, 2, size=(n, j, k), dtype=np.int8)
        arr2 = np.zeros((n, i, k), dtype=np.int8)

        target = furiosa.TargetAttr.get(npu=0, pe_begin=0, pe_end=0)
        execution_engine = ExecutionEngine(module, target)
        execution_engine.invoke("kernel", [arr0, arr1], [arr2])

        expected = np.einsum("nij,njk->nik", arr0, arr1)
        print(np.array_equal(arr2, expected))

test_compiler_high_level()
test_compiler_and_execution_engine()
test_minimum_dimension()
test_various_dimensions()
test_one_dimension()
