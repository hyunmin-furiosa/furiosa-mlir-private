from furiosa_mlir import *
from furiosa_mlir.ir import *
from furiosa_mlir.passmanager import *

class Compiler:
    def __init__(self, debug=False):
        self.debug = debug

    def apply_passes(self, module: Module, passes):
        for p in passes:
            pm = PassManager.parse(p)
            pm.run(module.operation)
            if self.debug:
                print("Applied pass: " + p)
                print(module)

    def apply_tosa_to_linalg_named_pass(self, module: Module):
        self.apply_passes(module, ["builtin.module(func.func(tosa-to-linalg-named))"])

    def apply_linalg_generalize_to_contract_ops_pass(self, module):
        self.apply_passes(module, ["builtin.module(linalg-generalize-to-contract-ops)"])

    def apply_transform_interpreter_pass(self, module: Module):
        self.apply_passes(module, ["builtin.module(transform-interpreter)"])

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
        if self.debug:
            print("Added transform module")
            print(module)
        self.apply_transform_interpreter_pass(module)
        module.body.operations[-1].erase()
        if self.debug:
            print("Removed transform module")
            print(module)

    def apply_transform_interpreter_pass(self, module: Module):
        self.apply_passes(module, ["builtin.module(transform-interpreter)"])

    def apply_convert_linalg_to_furiosa_pass(self, module: Module):
        self.apply_passes(module, ["builtin.module(convert-linalg-to-furiosa)"])

    def apply_furiosa_promote_slice_partition_loop_pass(self, module: Module):
        self.apply_passes(module, ["builtin.module(furiosa-promote-slice-partition-loop)"])

    def apply_furiosa_load_tensor_register_file_pass(self, module: Module):
        self.apply_passes(module, ["builtin.module(furiosa-load-tensor-register-file)"])

    def apply_func_results_to_params_pass(self, module: Module):
        self.apply_passes(module, ["builtin.module(func-results-to-params)"])

    def apply_furiosa_deallocation_pass(self, module: Module):
        self.apply_passes(module, ["builtin.module(furiosa-deallocation)"])

    def apply_optimize_allocation_liveness_pass(self, module: Module):
        self.apply_passes(module, ["builtin.module(func.func(optimize-allocation-liveness))"])

    def apply_furiosa_allocate_address_pass(self, module: Module):
        self.apply_passes(module, ["builtin.module(furiosa-allocate-address)"])

    def apply_convert_furiosa_to_furiosa_task_pass(self, module: Module):
        self.apply_passes(module, ["builtin.module(convert-furiosa-to-furiosa-task)"])

    def compile(self, module: Module):
        if self.debug:
            print("Compilation started")
            print(module)
        self.apply_tosa_to_linalg_named_pass(module)
        self.apply_linalg_generalize_to_contract_ops_pass(module)
        self.apply_transformations(module)
        self.apply_convert_linalg_to_furiosa_pass(module)
        self.apply_furiosa_promote_slice_partition_loop_pass(module)
        self.apply_furiosa_load_tensor_register_file_pass(module)
        self.apply_func_results_to_params_pass(module)
        self.apply_furiosa_deallocation_pass(module)
        self.apply_optimize_allocation_liveness_pass(module)
        self.apply_furiosa_allocate_address_pass(module)
        self.apply_convert_furiosa_to_furiosa_task_pass(module)
        if self.debug:
            print("Compilation ended")
        return module
