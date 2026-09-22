// RUN: not torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline %s 2>&1 | FileCheck %s

func.func @missing_moe_route_plugin(
    %router_logits: !torch.vtensor<[2,16],bf16>)
    -> (!torch.vtensor<[2,8],f32>, !torch.vtensor<[2,8],si32>) {
  %top_k = torch.constant.int 8
  %softmax = torch.constant.bool true
  // CHECK: error: 'torch.operator' op cannot lower to 'inceptron.moe_route' because it is not registered; load the Inceptron dialect plugin
  %weights, %ids = torch.operator "torch.inceptron.inceptron_moe_route"(
      %router_logits, %top_k, %softmax)
      : (!torch.vtensor<[2,16],bf16>, !torch.int, !torch.bool)
      -> (!torch.vtensor<[2,8],f32>, !torch.vtensor<[2,8],si32>)
  return %weights, %ids
      : !torch.vtensor<[2,8],f32>, !torch.vtensor<[2,8],si32>
}
