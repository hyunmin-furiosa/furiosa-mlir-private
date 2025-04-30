add_mlir_library(FuriosaMLIRExecutionEngine
  DeviceRuntime.cpp
  JitRunner.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/furiosa-mlir/ExecutionEngine
)

target_link_libraries(FuriosaMLIRExecutionEngine PRIVATE
  furiosa_torch
)
