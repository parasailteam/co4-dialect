// RUN: co4-opt %s | co4-opt --co4-lower --co4-linkbygpuid --co4-threadblockssa --co4-bufalloc | FileCheck %s

module {
    "co4hl.algo"() ({
      ^bb0 (  %g : tensor<4xf32>,
              %w : tensor<4xf32>,
              %m : tensor<1xf32>,
              %v : tensor<1xf32>,
              %const_beta1 : tensor<1xf32>,
              %const_compbeta1 : tensor<1xf32>,
              %const_invbeta1 : tensor<1xf32>,
              %const_beta2 : tensor<1xf32>,
              %const_compbeta2 : tensor<1xf32>,
              %const_invbeta2 : tensor<1xf32>,
              %const_lr : tensor<1xf32>) :
        %g1 = "co4hl.reduce_scatter"(%g) { func="add", dstbuf=4 } : (tensor<4xf32>) -> (tensor<1xf32>)
        // m1 = m * beta1 + g1 * compbeta1
        %tmp1 = std.mulf %m, %const_beta1                                          : tensor<1xf32>
        %tmp2 = std.mulf %g1, %const_compbeta1                                     : tensor<1xf32>
        %m1   = std.addf %tmp1, %tmp2              { dstbuf=5:i64 , dstoff=0:i64 } : tensor<1xf32>
        // v1 = v * beta2 + g1 * g1 * compbeta2
        %tmp3 = std.mulf %v, %const_beta2                                          : tensor<1xf32>
        %tmp4 = std.mulf %g1, %const_compbeta2                                     : tensor<1xf32>
        %tmp5 = std.mulf %g1, %tmp4                                                : tensor<1xf32>
        %v1   = std.addf %tmp3, %tmp5              { dstbuf=6:i64 , dstoff=0:i64 } : tensor<1xf32>
        // m_ = m1 / beta1
        %m_   = std.mulf %m1, %const_invbeta1      { dstbuf=8:i64 , dstoff=0:i64 } : tensor<1xf32>
        // v_ = v1 / beta2
        %v_   = std.mulf %v1, %const_invbeta2      { dstbuf=9:i64 , dstoff=0:i64 } : tensor<1xf32>
        // update = lr * m_ / sqrt(v_)
        %tmp7 = std.mulf %m_, %const_lr                                            : tensor<1xf32>
        %tmp8 = math.rsqrt %v_                                                     : tensor<1xf32>
        %scatteredUpdate = std.mulf %tmp7, %tmp8   { dstbuf=11:i64, dstoff=0:i64 } : tensor<1xf32>
        %update = "co4hl.all_gather"(%scatteredUpdate) { dstbuf=11 } : (tensor<1xf32>) -> (tensor<4xf32>)
        // Would be nice if type system indicated data was identical across ranks at this point
        // w1 = w - update
        %w1   = std.subf %w, %update               { dstbuf=7:i64 , dstoff=0:i64 } : tensor<4xf32>
        "co4hl.return"() : () -> ()
    }) { numgpus=4, numbufs=32, argbufs=[0,1,2,3,16,17,18,19,20,21,22]} : () -> ()



  // Implementations of collectives:


  module attributes{co4hl.collective="all_reduce"} {
  "co4ll.gpu"() ({
    %g1 = "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32> ) :
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
        "co4ll.return"(%g1) : (vector<4xf32>) -> ()
    }) : () -> (vector<4xf32>)
  }) { gpuid=0 } : () -> ()
  "co4ll.gpu"() ({
    %g1 = "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32> ) :
        // g1 = AllReduce(g)
        %g_0 = vector.extract_strided_slice %a0
            { offsets = [0], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_1 = vector.extract_strided_slice %a0
            { offsets = [1], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_2 = vector.extract_strided_slice %a0
            { offsets = [2], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_3 = vector.extract_strided_slice %a0
            { offsets = [3], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        "co4ll.send"(%g_0) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_3) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_2) : (vector<1xf32>) -> ()
        %g1_1 = "co4ll.rrc"(%g_1) : (vector<1xf32>) -> (vector<1xf32>)
        "co4ll.send"(%g1_1) : (vector<1xf32>) -> ()
        %g1_0 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_3 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_2 = "co4ll.recv"() : () -> (vector<1xf32>)
        %g1 = "co4ll.concat"(%g1_0, %g1_1, %g1_2, %g1_3) { dstbuf=4:i64 , dstoff=0:i64 } : (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>) -> (vector<4xf32>)
        "co4ll.return"(%g1) : (vector<4xf32>) -> ()
    }) : () -> (vector<4xf32>)
  }) { gpuid=1 } : () -> ()
  "co4ll.gpu"() ({
    %g1 = "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32> ) :
        // g1 = AllReduce(g)
        %g_0 = vector.extract_strided_slice %a0
            { offsets = [0], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_1 = vector.extract_strided_slice %a0
            { offsets = [1], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_2 = vector.extract_strided_slice %a0
            { offsets = [2], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_3 = vector.extract_strided_slice %a0
            { offsets = [3], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        "co4ll.send"(%g_1) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_0) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_3) : (vector<1xf32>) -> ()
        %g1_2 = "co4ll.rrc"(%g_2) : (vector<1xf32>) -> (vector<1xf32>)
        "co4ll.send"(%g1_2) : (vector<1xf32>) -> ()
        %g1_1 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_0 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_3 = "co4ll.recv"() : () -> (vector<1xf32>)
        %g1 = "co4ll.concat"(%g1_0, %g1_1, %g1_2, %g1_3) { dstbuf=4:i64 , dstoff=0:i64 } : (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>) -> (vector<4xf32>)
        "co4ll.return"(%g1) : (vector<4xf32>) -> ()
    }) : () -> (vector<4xf32>)
  }) { gpuid=2 } : () -> ()
  "co4ll.gpu"() ({
    %g1 = "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32> ) :
        // g1 = AllReduce(g)
        %g_0 = vector.extract_strided_slice %a0
            { offsets = [0], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_1 = vector.extract_strided_slice %a0
            { offsets = [1], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_2 = vector.extract_strided_slice %a0
            { offsets = [2], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_3 = vector.extract_strided_slice %a0
            { offsets = [3], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        "co4ll.send"(%g_2) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_1) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_0) : (vector<1xf32>) -> ()
        %g1_3 = "co4ll.rrc"(%g_3) : (vector<1xf32>) -> (vector<1xf32>)
        "co4ll.send"(%g1_3) : (vector<1xf32>) -> ()
        %g1_2 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_1 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_0 = "co4ll.recv"() : () -> (vector<1xf32>)
        %g1 = "co4ll.concat"(%g1_0, %g1_1, %g1_2, %g1_3) { dstbuf=4:i64 , dstoff=0:i64 } : (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>) -> (vector<4xf32>)
        "co4ll.return"(%g1) : (vector<4xf32>) -> ()
    }) : () -> (vector<4xf32>)
  }) { gpuid=3 } : () -> ()
  }


  module attributes{co4hl.collective="reduce_scatter"} {
  "co4ll.gpu"() ({
    %g1_0 = "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32> ) :
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
        "co4ll.return"(%g1_0) : (vector<1xf32>) -> ()
    }) : () -> (vector<1xf32>)
  }) { gpuid=0 } : () -> ()
  "co4ll.gpu"() ({
    %g1_0 = "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32> ) :
        %g_0 = vector.extract_strided_slice %a0
            { offsets = [0], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_1 = vector.extract_strided_slice %a0
            { offsets = [1], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_2 = vector.extract_strided_slice %a0
            { offsets = [2], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_3 = vector.extract_strided_slice %a0
            { offsets = [3], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        "co4ll.send"(%g_0) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_3) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_2) : (vector<1xf32>) -> ()
        %g1_1 = "co4ll.rrc"(%g_1) : (vector<1xf32>) -> (vector<1xf32>)
        "co4ll.return"(%g1_1) : (vector<1xf32>) -> ()
    }) : () -> (vector<1xf32>)
  }) { gpuid=1 } : () -> ()
  "co4ll.gpu"() ({
    %g1_2 = "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32> ) :
        %g_0 = vector.extract_strided_slice %a0
            { offsets = [0], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_1 = vector.extract_strided_slice %a0
            { offsets = [1], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_2 = vector.extract_strided_slice %a0
            { offsets = [2], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_3 = vector.extract_strided_slice %a0
            { offsets = [3], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        "co4ll.send"(%g_1) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_0) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_3) : (vector<1xf32>) -> ()
        %g1_2 = "co4ll.rrc"(%g_2) : (vector<1xf32>) -> (vector<1xf32>)
        "co4ll.return"(%g1_2) : (vector<1xf32>) -> ()
    }) : () -> (vector<1xf32>)
  }) { gpuid=2 } : () -> ()
  "co4ll.gpu"() ({
    %g1_3 = "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32> ) :
        %g_0 = vector.extract_strided_slice %a0
            { offsets = [0], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_1 = vector.extract_strided_slice %a0
            { offsets = [1], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_2 = vector.extract_strided_slice %a0
            { offsets = [2], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        %g_3 = vector.extract_strided_slice %a0
            { offsets = [3], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        "co4ll.send"(%g_2) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_1) : (vector<1xf32>) -> ()
        "co4ll.rrs"(%g_0) : (vector<1xf32>) -> ()
        %g1_3 = "co4ll.rrc"(%g_3) : (vector<1xf32>) -> (vector<1xf32>)
        "co4ll.return"(%g1_3) : (vector<1xf32>) -> ()
    }) : () -> (vector<1xf32>)
  }) { gpuid=3 } : () -> ()
  }


  module attributes{co4hl.collective="all_gather"} {
  "co4ll.gpu"() ({
    %g1 = "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32> ) :
        %g1_0 = vector.extract_strided_slice %a0
            { offsets = [0], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        "co4ll.send"(%g1_0) : (vector<1xf32>) -> ()
        %g1_3 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_2 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_1 = "co4ll.recv"() : () -> (vector<1xf32>)
        %g1 = "co4ll.concat"(%g1_0, %g1_1, %g1_2, %g1_3) { dstbuf=4:i64 , dstoff=0:i64 } : (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>) -> (vector<4xf32>)
        "co4ll.return"(%g1) : (vector<4xf32>) -> ()
    }) : () -> (vector<4xf32>)
  }) { gpuid=0 } : () -> ()
  "co4ll.gpu"() ({
    %g1 = "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32> ) :
        %g1_1 = vector.extract_strided_slice %a0
            { offsets = [1], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        "co4ll.send"(%g1_1) : (vector<1xf32>) -> ()
        %g1_0 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_3 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_2 = "co4ll.recv"() : () -> (vector<1xf32>)
        %g1 = "co4ll.concat"(%g1_0, %g1_1, %g1_2, %g1_3) { dstbuf=4:i64 , dstoff=0:i64 } : (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>) -> (vector<4xf32>)
        "co4ll.return"(%g1) : (vector<4xf32>) -> ()
    }) : () -> (vector<4xf32>)
  }) { gpuid=1 } : () -> ()
  "co4ll.gpu"() ({
    %g1 = "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32> ) :
        %g1_2 = vector.extract_strided_slice %a0
            { offsets = [2], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        "co4ll.send"(%g1_2) : (vector<1xf32>) -> ()
        %g1_1 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_0 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_3 = "co4ll.recv"() : () -> (vector<1xf32>)
        %g1 = "co4ll.concat"(%g1_0, %g1_1, %g1_2, %g1_3) { dstbuf=4:i64 , dstoff=0:i64 } : (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>) -> (vector<4xf32>)
        "co4ll.return"(%g1) : (vector<4xf32>) -> ()
    }) : () -> (vector<4xf32>)
  }) { gpuid=2 } : () -> ()
  "co4ll.gpu"() ({
    %g1 = "co4ll.tb"() ({
      ^bb0 (  %a0 : vector<4xf32> ) :
        %g1_3 = vector.extract_strided_slice %a0
            { offsets = [3], sizes = [1], strides = [1] } : vector<4xf32> to vector<1xf32>
        "co4ll.send"(%g1_3) : (vector<1xf32>) -> ()
        %g1_2 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_1 = "co4ll.rcs"() : () -> (vector<1xf32>)
        %g1_0 = "co4ll.recv"() : () -> (vector<1xf32>)
        %g1 = "co4ll.concat"(%g1_0, %g1_1, %g1_2, %g1_3) { dstbuf=4:i64 , dstoff=0:i64 } : (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>) -> (vector<4xf32>)
        "co4ll.return"(%g1) : (vector<4xf32>) -> ()
    }) : () -> (vector<4xf32>)
  }) { gpuid=3 } : () -> ()
  }
    // CHECK-LABEL: co4ll.gpu
    // CHECK-LABEL: co4ll.gpu
    // CHECK-LABEL: co4ll.gpu
    // CHECK-LABEL: co4ll.gpu
}
