add_mlir_library(FuriosaMLIRExecutionEngine
  ExecutionEngine.cpp
  JitRunner.cpp
  RenegadeRuntime.cpp

  DEPENDS
  FuriosaMLIRTargetC
)

target_link_libraries(FuriosaMLIRExecutionEngine PRIVATE
  device_runtime
)
