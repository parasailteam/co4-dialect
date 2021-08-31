// RUN: co4-opt %s | co4-opt | FileCheck %s

module {
    // CHECK-LABEL: co4hl.algo
    "co4hl.algo"() ({
      ^bb0 (  %g : tensor<?xf32>,
              %w : tensor<?xf32>,
              %m : tensor<?xf32>,
              %v : tensor<?xf32>,
              %const_beta1 : tensor<?xf32>,
              %const_compbeta1 : tensor<?xf32>,
              %const_invbeta1 : tensor<?xf32>,
              %const_beta2 : tensor<?xf32>,
              %const_compbeta2 : tensor<?xf32>,
              %const_invbeta2 : tensor<?xf32>,
              %const_lr : tensor<?xf32>) :
        %g1 = "co4hl.reduce_scatter"(%g) { func="add", dstbuf=4 } : (tensor<?xf32>) -> (tensor<?xf32>)
        // m1 = m * beta1 + g1 * compbeta1
        %tmp1 = std.mulf %m, %const_beta1                                          : tensor<?xf32>
        %tmp2 = std.mulf %g1, %const_compbeta1                                     : tensor<?xf32>
        %m1   = std.addf %tmp1, %tmp2              { dstbuf=5:i64 , dstoff=0:i64 } : tensor<?xf32>
        // v1 = v * beta2 + g1 * g1 * compbeta2
        %tmp3 = std.mulf %v, %const_beta2                                          : tensor<?xf32>
        %tmp4 = std.mulf %g1, %const_compbeta2                                     : tensor<?xf32>
        %tmp5 = std.mulf %g1, %tmp4                                                : tensor<?xf32>
        %v1   = std.addf %tmp3, %tmp5              { dstbuf=6:i64 , dstoff=0:i64 } : tensor<?xf32>
        // m_ = m1 / beta1
        %m_   = std.mulf %m1, %const_invbeta1      { dstbuf=8:i64 , dstoff=0:i64 } : tensor<?xf32>
        // v_ = v1 / beta2
        %v_   = std.mulf %v1, %const_invbeta2      { dstbuf=9:i64 , dstoff=0:i64 } : tensor<?xf32>
        // update = lr * m_ / sqrt(v_)
        %tmp7 = std.mulf %m_, %const_lr                                            : tensor<?xf32>
        %tmp8 = math.rsqrt %v_                                                     : tensor<?xf32>
        %scatteredUpdate = std.mulf %tmp7, %tmp8   { dstbuf=11:i64, dstoff=0:i64 } : tensor<?xf32>
        %update = "co4hl.all_gather"(%scatteredUpdate) { dstbuf=11 } : (tensor<?xf32>) -> (tensor<?xf32>)
        // Would be nice if type system indicated data was identical across ranks at this point
        // w1 = w - update
        %w1   = std.subf %w, %update               { dstbuf=7:i64 , dstoff=0:i64 } : tensor<?xf32>
        "co4hl.return"() : () -> ()
    }) { numgpus=4, numbufs=32, argbufs=[0,1,2,3,16,17,18,19,20,21,22]} : () -> ()
}
