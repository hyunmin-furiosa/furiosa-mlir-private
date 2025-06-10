include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=furiosa_mlir.")

################################################################################
# Structural groupings.
################################################################################

declare_mlir_python_sources(FuriosaMLIRPythonSources)
declare_mlir_python_sources(FuriosaMLIRPythonSources.Dialects
  ADD_TO_PARENT FuriosaMLIRPythonSources)

################################################################################
# Pure python sources and generated code
################################################################################

declare_mlir_python_sources(FuriosaMLIRPythonCAPI.HeaderSources
  ROOT_DIR "${FURIOSA_MLIR_SOURCE_DIR}/include"
  SOURCES_GLOB "furiosa-mlir-c/*.h"
)

################################################################################
# Dialect bindings
################################################################################

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT FuriosaMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/furiosa-mlir"
  TD_FILE dialects/FuriosaOps.td
  SOURCES
    dialects/furiosa.py
  DIALECT_NAME furiosa
  GEN_ENUM_BINDINGS)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT FuriosaMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/furiosa-mlir"
  TD_FILE dialects/HostOps.td
  SOURCES
    dialects/host.py
  DIALECT_NAME furiosa_host
  GEN_ENUM_BINDINGS)

################################################################################
# Python extensions.
# The sources for these are all in lib/Bindings/Python, but since they have to
# be rebuilt for each package and integrate with the source setup here, we
# just reference them here instead of having ordered, cross package target
# dependencies.
################################################################################

set(PYTHON_SOURCE_DIR "${FURIOSA_MLIR_SOURCE_DIR}/lib/Bindings/Python")

declare_mlir_python_extension(FuriosaMLIRPythonExtension.RegisterEverything
  MODULE_NAME _mlirRegisterEverything
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    RegisterEverything.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIConversion
    MLIRCAPITransforms
    FuriosaMLIRCAPIRegisterEverything
)

declare_mlir_python_extension(FuriosaMLIRPythonExtension.Dialects.Furiosa.Pybind
  MODULE_NAME _furiosaMlirDialectsFuriosa
  ADD_TO_PARENT FuriosaMLIRPythonSources.Dialects.furiosa
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    DialectFuriosa.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    FuriosaMLIRCAPIFuriosa
)

declare_mlir_python_extension(FuriosaMLIRPythonExtension.Dialects.Host.Pybind
  MODULE_NAME _furiosaMlirDialectsHost
  ADD_TO_PARENT FuriosaMLIRPythonSources.Dialects.furiosa_host
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    DialectHost.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    FuriosaMLIRCAPIFuriosa
)

declare_mlir_python_extension(FuriosaMLIRPythonExtension.FuriosaDialectPasses
  MODULE_NAME _furiosaMlirFuriosaPasses
  ADD_TO_PARENT FuriosaMLIRPythonSources.Dialects.furiosa
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    FuriosaPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    FuriosaMLIRCAPIFuriosa
)

################################################################################
# Common CAPI dependency DSO.
# All python extensions must link through one DSO which exports the CAPI, and
# this must have a globally unique name amongst all embeddors of the python
# library since it will effectively have global scope.
#
# The presence of this aggregate library is part of the long term plan, but its
# use needs to be made more flexible.
#
# TODO: Upgrade to the aggregate utility in https://reviews.llvm.org/D106419
# once ready.
################################################################################

add_mlir_python_common_capi_library(FuriosaMLIRPythonCAPI
  INSTALL_COMPONENT FuriosaMLIRPythonModules
  INSTALL_DESTINATION python_packages/furiosa-mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python_packages/furiosa-mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT ".."
  DECLARED_HEADERS
    FuriosaMLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    MLIRPythonSources
    FuriosaMLIRPythonSources
    FuriosaMLIRPythonExtension.RegisterEverything
)

################################################################################
# The fully assembled package of modules.
# This must come last.
################################################################################

add_mlir_python_modules(FuriosaMLIRPythonModules
  ROOT_PREFIX "${FURIOSA_MLIR_BINARY_DIR}/python_packages/furiosa_mlir"
  INSTALL_PREFIX "python_packages/furiosa-mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    FuriosaMLIRPythonSources
    FuriosaMLIRPythonExtension.RegisterEverything
  COMMON_CAPI_LINK_LIBS
    FuriosaMLIRPythonCAPI
)
