// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = co4ll.foo %{{.*}} : i32
        %res = co4ll.foo %0 : i32
        return
    }
}
