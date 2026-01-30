// RUN: torch-mlir-opt --torch-lower-inceptron-ops --split-input-file %s | FileCheck %s

// CHECK: func.func @netcrush_scaled_mm(%arg0: !torch.tensor<[64,128],f32>, %arg1: !torch.tensor<[128,32],f32>, %arg2: !torch.tensor, %arg3: !torch.tensor, %arg4: !torch.tensor) -> !torch.tensor<[64,32],f32>
// CHECK-LABEL: func.func @main(
// CHECK:   %[[CALL:.*]] = func.call @netcrush_scaled_mm(%arg0, %arg1, %arg2, %arg3, %arg4)
// CHECK:   return %[[CALL]] : !torch.tensor<[64,32],f32>
module {
  func.func @main(%arg0: !torch.tensor<[64,128],f32>,
                  %arg1: !torch.tensor<[128,32],f32>,
                  %arg2: !torch.tensor,
                  %arg3: !torch.tensor,
                  %arg4: !torch.tensor) -> !torch.tensor<[64,32],f32> {
    %0 = torch.operator "netcrush_ext.netcrush_scaled_mm"(%arg0, %arg1, %arg2, %arg3, %arg4) :
      (!torch.tensor<[64,128],f32>, !torch.tensor<[128,32],f32>, !torch.tensor, !torch.tensor, !torch.tensor) ->
      !torch.tensor<[64,32],f32>
    return %0 : !torch.tensor<[64,32],f32>
  }
}
