# RUN: %PYTHON

from furiosa_mlir.ir import *
from furiosa_mlir.passmanager import *

import furiosa_mlir.dialects.furiosa as furiosa
import furiosa_mlir.dialects.host as furiosa_host

def testType():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        device_type = furiosa_host.DeviceType.get()
        hal_program_type = furiosa_host.HalProgramType.get()
    
    print(device_type)
    print(hal_program_type)

def testAttribute():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        target_attr = furiosa.TargetAttr.get(npu=0, pe_begin=0, pe_end=0)
    
    print(target_attr)

def testOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            target_attr = furiosa.TargetAttr.get(npu=0, pe_begin=0, pe_end=0)
            device_new_op = furiosa_host.DeviceNewOp(target_attr)
            hal_program = furiosa_host.HalProgramSeqOp([])
            device_execute_op = furiosa_host.DeviceExecuteOp(
                device_new_op, hal_program)

    print(device_new_op)
    print(hal_program)
    print(device_execute_op)

testType()
testAttribute()
testOp()
