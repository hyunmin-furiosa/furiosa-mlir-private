from furiosa_mlir import *
from furiosa_mlir.ir import *
from furiosa_mlir.passmanager import *

class Compiler:
    def apply_tosa_to_linalg_named_pass(self, module: Module):
        pm = PassManager.parse("builtin.module(func.func(tosa-to-linalg-named))")
        pm.run(module.operation)

    def apply_linalg_generalize_to_contract_ops_pass(self, module):
        pm = PassManager.parse("builtin.module(linalg-generalize-to-contract-ops)")
        pm.run(module.operation)

    def apply_transform_interpreter_pass(self, module: Module):
        pm = PassManager.parse("builtin.module(transform-interpreter)")
        pm.run(module.operation)

    def apply_transformations(self, module: Module):
        transform_module = Module.parse(
r"""
module {
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:1 = transform.structured.tile_using_forall %0 num_threads [64] { mapping = [ #furiosa.mapping ] } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}
"""
        )
        module.body.append(transform_module.body.operations[0])
        self.apply_transform_interpreter_pass(module)
        module.body.operations[-1].erase()

    def apply_transform_interpreter_pass(self, module: Module):
        pm = PassManager.parse("builtin.module(transform-interpreter)")
        pm.run(module.operation)

    def apply_convert_linalg_to_furiosa_pass(self, module: Module):
        pm = PassManager.parse("builtin.module(convert-linalg-to-furiosa)")
        pm.run(module.operation)

    def apply_furiosa_promote_slice_partition_loop_pass(self, module: Module):
        pm = PassManager.parse("builtin.module(furiosa-promote-slice-partition-loop)")
        pm.run(module.operation)

    def apply_func_results_to_params_pass(self, module: Module):
        pm = PassManager.parse("builtin.module(func-results-to-params)")
        pm.run(module.operation)

    def apply_furiosa_deallocation_pass(self, module: Module):
        pm = PassManager.parse("builtin.module(furiosa-deallocation)")
        pm.run(module.operation)

    def apply_optimize_allocation_liveness_pass(self, module: Module):
        pm = PassManager.parse("builtin.module(func.func(optimize-allocation-liveness))")
        pm.run(module.operation)

    def apply_furiosa_allocate_address_pass(self, module: Module):
        pm = PassManager.parse("builtin.module(furiosa-allocate-address)")
        pm.run(module.operation)

    def apply_convert_furiosa_to_furiosa_task_pass(self, module: Module):
        pm = PassManager.parse("builtin.module(convert-furiosa-to-furiosa-task)")
        pm.run(module.operation)

    def compile(self, module: Module):
        self.apply_tosa_to_linalg_named_pass(module)
        self.apply_linalg_generalize_to_contract_ops_pass(module)
        self.apply_transformations(module)
        self.apply_convert_linalg_to_furiosa_pass(module)
        self.apply_furiosa_promote_slice_partition_loop_pass(module)
        self.apply_func_results_to_params_pass(module)
        self.apply_furiosa_deallocation_pass(module)
        self.apply_optimize_allocation_liveness_pass(module)
        self.apply_furiosa_allocate_address_pass(module)
        self.apply_convert_furiosa_to_furiosa_task_pass(module)
        return module
