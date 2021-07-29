# Out-of-tree dialects for MLIR

This repository contains out-of-tree [MLIR](https://mlir.llvm.org/) dialects as well as a
standalone `opt`-like tool to operate on those dialects.

## How to build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-co4-opt
```
To build the documentation from the TableGen description of the dialects
operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.

## License

These dialects are made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.
