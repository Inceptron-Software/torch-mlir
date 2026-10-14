# RUN: %PYTHON %s | FileCheck %s

"""Regression coverage for specialized scalar operands in symbolic FX arithmetic."""

import operator

import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch_mlir.fx import stateless_fx_import


def test_symbolic_arithmetic_accepts_specialized_scalar_nodes(
    specialized: bool,
) -> None:
    shape_env = ShapeEnv()
    graph = torch.fx.Graph()
    batch = graph.placeholder("batch")
    batch.meta["val"] = 8 if specialized else shape_env.create_unbacked_symint()
    product = graph.call_function(operator.mul, (32, batch))
    # Decomposition can retain a symbolic result after specializing its operand.
    product.meta["val"] = shape_env.create_unbacked_symint()
    graph.output((product,))
    module = torch.fx.GraphModule(torch.nn.Module(), graph)
    mlir = stateless_fx_import(module, output_type="raw")
    assert mlir.operation.verify()
    print(mlir)


# CHECK-LABEL: symbolic_scalar
# CHECK: torch.aten.mul.int
print("symbolic_scalar")
test_symbolic_arithmetic_accepts_specialized_scalar_nodes(False)

# CHECK-LABEL: specialized_scalar
# CHECK: torch.aten.mul.int
print("specialized_scalar")
test_symbolic_arithmetic_accepts_specialized_scalar_nodes(True)
