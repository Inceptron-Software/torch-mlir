// RUN: torch-mlir-opt --torch-lower-inceptron-ops --split-input-file %s | FileCheck %s

// CHECK: func.func private @inceptron_scaled_mm_ret(!torch.tensor<[64,128],f32>, !torch.tensor<[128,32],f32>, !torch.tensor, !torch.tensor, !torch.tensor) -> !torch.tensor<[64,32],f32>
// CHECK-LABEL: func.func @main(
// CHECK:   %[[CALL:.*]] = call @inceptron_scaled_mm_ret(%arg0, %arg1, %arg2, %arg3, %arg4)
// CHECK:   return %[[CALL]] : !torch.tensor<[64,32],f32>
module {
  func.func @main(%arg0: !torch.tensor<[64,128],f32>,
                  %arg1: !torch.tensor<[128,32],f32>,
                  %arg2: !torch.tensor,
                  %arg3: !torch.tensor,
                  %arg4: !torch.tensor) -> !torch.tensor<[64,32],f32> {
    %none = torch.constant.none
    %0 = torch.operator "inceptron.inceptron_scaled_mm"(%arg0, %arg1, %arg2, %arg3, %arg4, %none) :
      (!torch.tensor<[64,128],f32>, !torch.tensor<[128,32],f32>, !torch.tensor, !torch.tensor, !torch.tensor, !torch.none) ->
      !torch.tensor<[64,32],f32>
    return %0 : !torch.tensor<[64,32],f32>
  }
}

// -----

// CHECK: func.func private @inceptron_scaled_mm_64_128_128_32_ret_64_32(!torch.vtensor<[64,128],f8E4M3FN>, !torch.vtensor<[128,32],f8E4M3FN>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[64,32],bf16>
// CHECK-LABEL: func.func @main_aten_scaled_mm(
// CHECK:   %[[CALL:.*]] = call @inceptron_scaled_mm_64_128_128_32_ret_64_32(%arg0, %arg1, %arg2, %arg3, %int15)
// CHECK:   return %[[CALL]] : !torch.vtensor<[64,32],bf16>
module {
  func.func @main_aten_scaled_mm(%arg0: !torch.vtensor<[64,128],f8E4M3FN>,
                                 %arg1: !torch.vtensor<[128,32],f8E4M3FN>,
                                 %arg2: !torch.vtensor<[],f32>,
                                 %arg3: !torch.vtensor<[],f32>) -> !torch.vtensor<[64,32],bf16> {
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %false = torch.constant.bool false
    %0 = torch.operator "torch.aten._scaled_mm"(%arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %false) :
      (!torch.vtensor<[64,128],f8E4M3FN>, !torch.vtensor<[128,32],f8E4M3FN>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.bool) ->
      !torch.vtensor<[64,32],bf16>
    return %0 : !torch.vtensor<[64,32],bf16>
  }
}
