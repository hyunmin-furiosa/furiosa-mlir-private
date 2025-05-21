add_mlir_library(FuriosaMLIRExecutionEngine
  DeviceRuntime.cpp
  JitRunner.cpp

  DEPENDS
  FuriosaMLIRTargetC
)

target_link_libraries(FuriosaMLIRExecutionEngine PRIVATE
  furiosa_torch
)
