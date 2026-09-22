// RUN: not torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline %s 2>&1 | FileCheck %s

func.func @unsupported(%input: !torch.vtensor<[2,3],f32>)
    -> !torch.vtensor<[2,3],f32> {
  // CHECK: error: 'torch.operator' op unsupported Inceptron custom operator 'torch.inceptron.inceptron_unknown'
  %result = torch.operator "torch.inceptron.inceptron_unknown"(%input)
      : (!torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32>
  return %result : !torch.vtensor<[2,3],f32>
}
