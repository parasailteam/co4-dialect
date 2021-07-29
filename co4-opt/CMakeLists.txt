get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRCo4LL
        )
add_llvm_executable(co4-opt co4-opt.cpp)

llvm_update_compile_flags(co4-opt)
target_link_libraries(co4-opt PRIVATE ${LIBS})
