// RUN: co4-opt %s | co4-opt --co4-bufalloc | co4-opt | FileCheck %s

module {
  "co4ll.gpu"() ({
    // CHECK-LABEL: co4ll.tb
    "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32>,
              %a1 : vector<4xf32>,
              %a2 : vector<4xf32>,
              %a3 : vector<4xf32>,
              %a4 : vector<4xf32>,
              %a5 : vector<4xf32>,
              %a6 : vector<4xf32>,
              %a7 : vector<4xf32>,
              %a8 : vector<4xf32>,
              %a9 : vector<4xf32>,
              %a10 : vector<4xf32>,
              %a11 : vector<4xf32>,
              %a12 : vector<4xf32>,
              %a13 : vector<4xf32>,
              %a14 : vector<4xf32>,
              %a15 : vector<4xf32>,
              %a16 : vector<4xf32>,
              %a17 : vector<4xf32>,
              %a18 : vector<4xf32>,
              %a19 : vector<4xf32>,
              %a20 : vector<4xf32>,
              %a21 : vector<4xf32>,
              %a22 : vector<4xf32>,
              %a23 : vector<4xf32>,
              %a24 : vector<4xf32>,
              %a25 : vector<4xf32>,
              %a26 : vector<4xf32>,
              %a27 : vector<4xf32>,
              %a28 : vector<4xf32>,
              %a29 : vector<4xf32>,
              %a30 : vector<4xf32>,
              %a31 : vector<4xf32>) :
        // g1 = AllReduce(g)
        %g_0 = vector.extract_strided_slice %a0
            { offsets = [0], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_1 = vector.extract_strided_slice %a0
            { offsets = [1], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_2 = vector.extract_strided_slice %a0
            { offsets = [2], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_3 = vector.extract_strided_slice %a0
            { offsets = [3], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        "co4ll.send"(%g_3) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_2) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_1) : (vector<1xf32>) -> ()
        %g1_0 = "co4ll.rrc"(%g_0) : (vector<1xf32>) -> (vector<1xf32>)
        "co4ll.send"(%g1_0) : (vector<1xf32>) -> ()
        %g1_3 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_2 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_1 = "co4ll.recv"() : () -> (vector<1xf32>)
        %g1 = "co4ll.concat"(%g1_0, %g1_1, %g1_2, %g1_3) { dstbuf=4:i64 , dstoff=0:i64 } : (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>) -> (vector<4xf32>)
        "co4ll.return"() : () -> ()
    }) : () -> ()
  }) { gpuid=0 } : () -> ()
  "co4ll.gpu"() ({
    // CHECK-LABEL: co4ll.tb
    %w1 = "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32>,
              %a1 : vector<4xf32>,
              %a2 : vector<4xf32>,
              %a3 : vector<4xf32>,
              %a4 : vector<4xf32>,
              %a5 : vector<4xf32>,
              %a6 : vector<4xf32>,
              %a7 : vector<4xf32>,
              %a8 : vector<4xf32>,
              %a9 : vector<4xf32>,
              %a10 : vector<4xf32>,
              %a11 : vector<4xf32>,
              %a12 : vector<4xf32>,
              %a13 : vector<4xf32>,
              %a14 : vector<4xf32>,
              %a15 : vector<4xf32>,
              %a16 : vector<4xf32>,
              %a17 : vector<4xf32>,
              %a18 : vector<4xf32>,
              %a19 : vector<4xf32>,
              %a20 : vector<4xf32>,
              %a21 : vector<4xf32>,
              %a22 : vector<4xf32>,
              %a23 : vector<4xf32>,
              %a24 : vector<4xf32>,
              %a25 : vector<4xf32>,
              %a26 : vector<4xf32>,
              %a27 : vector<4xf32>,
              %a28 : vector<4xf32>,
              %a29 : vector<4xf32>,
              %a30 : vector<4xf32>,
              %a31 : vector<4xf32>) :
        // m1 = m * beta1 + g1 * compbeta1
        %tmp1 = std.mulf %a2, %a16                                                                  : vector<4xf32>
        %tmp2 = std.mulf %a4, %a17                                                                  : vector<4xf32>
        %m1   = std.addf %tmp1, %tmp2              { dstbuf=5:i64 , dstoff=0:i64 } : vector<4xf32>
        // v1 = v * beta2 + g1 * g1 * compbeta2
        %tmp3 = std.mulf %a3, %a19                                                                  : vector<4xf32>
        %tmp4 = std.mulf %a4, %a20                                                                  : vector<4xf32>
        %tmp5 = std.mulf %a4, %tmp4                                                                 : vector<4xf32>
        %v1   = std.addf %tmp3, %tmp5              { dstbuf=6:i64 , dstoff=0:i64 } : vector<4xf32>
        // m_ = m1 / beta1
        %m_   = std.mulf %m1, %a18                 { dstbuf=8:i64 , dstoff=0:i64 } : vector<4xf32>
        // v_ = v1 / beta2
        %v_   = std.mulf %v1, %a21                 { dstbuf=9:i64 , dstoff=0:i64 } : vector<4xf32>
        // update = lr * m_ / sqrt(v_)
        %scatteredUpdate = std.mulf %m_, %a22      { dstbuf=10:i64, dstoff=0:i64 } : vector<4xf32>
        %tmp8 = math.rsqrt %v_                                                                      : vector<4xf32>
        %update = std.mulf %scatteredUpdate, %tmp8 { dstbuf=11:i64, dstoff=0:i64 } : vector<4xf32>
        // w1 = w - update
        %w1   = std.subf %a1, %update              { dstbuf=7:i64 , dstoff=0:i64 } : vector<4xf32>
        "co4ll.return"(%w1) : (vector<4xf32>) -> ()
    }) : () -> (vector<4xf32>)
  }) { gpuid=0 } : () -> ()
}
