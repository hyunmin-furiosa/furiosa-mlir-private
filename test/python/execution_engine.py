from furiosa_mlir import *
from furiosa_mlir.execution_engine import *
from furiosa_mlir.ir import *
from furiosa_mlir.passmanager import *

def test_execution_engine():
    with Context() as ctx, Location.unknown():
        module = Module.parse(
r"""
func.func @add(%arg0: f32, %arg1: f32) -> f32 attributes { llvm.emit_c_interface } {
  %add = arith.addf %arg0, %arg1 : f32
  return %add : f32
}
"""
        )

        execution_engine = ExecutionEngine(module)
        execution_engine.invoke("add")

test_execution_engine()
