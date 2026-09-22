// RUN: not torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline %s 2>&1 | FileCheck %s

func.func @missing_plugin(
    %input: !torch.vtensor<[2,3],f32>, %communicator: !torch.int)
    -> !torch.vtensor<[2,3],f32> {
  // CHECK: error: 'torch.operator' op cannot lower to 'inceptron.all_reduce' because it is not registered; load the Inceptron dialect plugin
  %result = torch.operator "torch.inceptron.inceptron_all_reduce"(
      %input, %communicator)
      : (!torch.vtensor<[2,3],f32>, !torch.int)
      -> !torch.vtensor<[2,3],f32>
  return %result : !torch.vtensor<[2,3],f32>
}
