add_mlir_library(FuriosaMLIRExecutionEngine
  RenegadeRuntime.cpp
  JitRunner.cpp

  DEPENDS
  FuriosaMLIRTargetC
)

target_link_libraries(FuriosaMLIRExecutionEngine PRIVATE
  device_runtime
)
