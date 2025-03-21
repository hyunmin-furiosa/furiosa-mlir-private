add_mlir_library(FuriosaMLIRExecutionEngine
    DeviceRuntime.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/furiosa-mlir/ExecutionEngine
)

mlir_target_link_libraries(FuriosaMLIRExecutionEngine PUBLIC
    device_runtime
)
