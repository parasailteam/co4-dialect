// RUN: co4-opt %s | co4-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = co4hl.foo %{{.*}} : i32
        %res = co4hl.foo %0 : i32
        return
    }
}
