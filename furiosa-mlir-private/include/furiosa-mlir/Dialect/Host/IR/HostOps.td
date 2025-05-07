//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef HOST_OPS_TD
#define HOST_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaTypes.td"
include "furiosa-mlir/Dialect/Host/IR/HostBase.td"
include "furiosa-mlir/Dialect/Host/IR/HostTypes.td"

//===----------------------------------------------------------------------===//
// Host operations.
//===----------------------------------------------------------------------===//

def AllocOp : HostOp<"alloc"> {
  let summary = "Host memory allocation";
  let description = [{}];
  let arguments = (ins
      I64Attr:$size,
      OptionalAttr<I64ArrayAttr>:$data
  );
  let results = (outs
      BufferType:$buffer
  );
  let assemblyFormat = "attr-dict";
}

def FuncAllocOp : HostOp<"func_alloc"> {
  let summary = "Host memory allocation for kernel function";
  let description = [{}];
  let arguments = (ins
      FlatSymbolRefAttr:$function
  );
  let results = (outs
      BufferType:$buffer
  );
  let assemblyFormat = "attr-dict";
}

def CompareOp : HostOp<"compare"> {
  let summary = "Host tensor compare";
  let description = [{}];
  let arguments = (ins
      BufferType:$buffer0,
      BufferType:$buffer1
  );
  let results = (outs
      I1:$result
  );
  let assemblyFormat = "$buffer0 $buffer1 attr-dict";
}

//===----------------------------------------------------------------------===//
// Device runtime compatible operations.
//===----------------------------------------------------------------------===//

def PeProgramLoadInstOp : HostOp<"pe_program_load_inst"> {
  let summary = "PE program load instruction";
  let description = [{}];
  let arguments = (ins
      I64Attr:$dram_address,
      I64Attr:$spm_address,
      BufferType:$binary
  );
  let results = (outs
      PeProgramType:$pe_program
  );
  let assemblyFormat = "$binary attr-dict";
}

def PeProgramLaunchOp : HostOp<"pe_program_launch"> {
  let summary = "PE program launch";
  let description = [{}];
  let arguments = (ins
      I64Attr:$spm_address
  );
  let results = (outs
      PeProgramType:$pe_program
  );
  let assemblyFormat = "attr-dict";
}

def PeProgramSeqOp : HostOp<"pe_program_seq"> {
  let summary = "PE program sequence";
  let description = [{}];
  let arguments = (ins
      Variadic<PeProgramType>:$pe_programs
  );
  let results = (outs
      PeProgramType:$pe_program
  );
  let assemblyFormat = "$pe_programs attr-dict";
}

def HalProgramWriteAtOp : HostOp<"hal_program_write_at"> {
  let summary = "HAL program write at";
  let description = [{}];
  let arguments = (ins
      I64Attr:$dram_address,
      BufferType:$buffer
  );
  let results = (outs
      HalProgramType:$hal_program
  );
  let assemblyFormat = "$buffer attr-dict";
}

def HalProgramReadAtOp : HostOp<"hal_program_read_at"> {
  let summary = "HAL program read at";
  let description = [{}];
  let arguments = (ins
      I64Attr:$dram_address,
      BufferType:$buffer
  );
  let results = (outs
      HalProgramType:$hal_program
  );
  let assemblyFormat = "$buffer attr-dict";
}

def HalProgramExecuteOp : HostOp<"hal_program_execute"> {
  let summary = "HAL program execute";
  let description = [{}];
  let arguments = (ins
      PeProgramType:$pe_program
  );
  let results = (outs
      HalProgramType:$hal_program
  );
  let assemblyFormat = "$pe_program attr-dict";
}

def HalProgramSeqOp : HostOp<"hal_program_seq"> {
  let summary = "HAL program sequence";
  let description = [{}];
  let arguments = (ins
      Variadic<HalProgramType>:$hal_programs
  );
  let results = (outs
      HalProgramType:$hal_program
  );
  let assemblyFormat = "$hal_programs attr-dict";
}

def DeviceNewOp : HostOp<"device_new"> {
  let summary = "Device new";
  let description = [{}];
  let arguments = (ins
      TargetAttr:$target
  );
  let results = (outs
      DeviceType:$device
  );
  let assemblyFormat = "attr-dict";
}

def DeviceExecuteOp : HostOp<"device_execute"> {
  let summary = "Device execute";
  let description = [{}];
  let arguments = (ins
      DeviceType:$device,
      HalProgramType:$hal_program
  );
  let results = (outs
      ExecutionType:$execution
  );
  let assemblyFormat = "$device $hal_program attr-dict";
}

def DeviceExecutionWaitOp : HostOp<"device_execution_wait"> {
  let summary = "Device execution wait";
  let description = [{}];
  let arguments = (ins
      ExecutionType:$execution
  );
  let results = (outs);
  let assemblyFormat = "$execution attr-dict";
}

#endif // HOST_OPS_TD
