#pragma once

#include <tuple>
#include "mlir/IR/Value.h"

namespace mlir {
namespace co4ll {
std::tuple<int, int> getDstBufferAndOffset(const Value v);
}
}
