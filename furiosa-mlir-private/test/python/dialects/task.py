# RUN: %PYTHON

from furiosa_mlir.ir import *
from furiosa_mlir.passmanager import *

import furiosa_mlir.dialects.task as furiosa_task

def testType():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        sfr_type = furiosa_task.SfrType.get()
    
    print(sfr_type)

def testOp():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        flush_op = furiosa_task.PrflushOp([])

    print(flush_op)

testType()
testOp()
