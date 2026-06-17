/*
 * Inceptron AB ("COMPANY") CONFIDENTIAL
 * Unpublished Copyright (c) 2026.
 * All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains the property of
 * COMPANY. The intellectual and technical concepts contained herein are
 * proprietary to COMPANY and may be covered by U.S. and Foreign Patents,
 * patents in process, and are protected by trade secret or copyright law.
 * Dissemination of this information or reproduction of this material is
 * strictly forbidden unless prior written permission is obtained from COMPANY.
 * Access to the source code contained herein is hereby forbidden to anyone
 * except current COMPANY employees, managers or contractors who have executed
 * Confidentiality and Non-disclosure agreements explicitly covering such
 * access.
 *
 * The copyright notice above does not evidence any actual or intended
 * publication or disclosure of this source code, which includes information
 * that is confidential and/or proprietary, and is a trade secret, of COMPANY.
 * ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC  PERFORMANCE, OR PUBLIC
 * DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
 * CONSENT OF COMPANY IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
 * LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
 * CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
 * REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
 * SELL ANYTHING THAT IT MAY DESCRIBE, IN WHOLE OR IN PART.
 */

#include "PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/iterator_range.h"
#include <cstdint>
#include <string>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

constexpr StringLiteral kInceptronPrefix("inceptron.inceptron_");

/// Rewrites Inceptron custom torch.operators into a direct func.call to an
/// external function declaration. The declaration is created on-demand in the
/// module with a signature that matches the specific operator instance.
///
/// Handles:
/// - inceptron.inceptron_scaled_mm
/// - inceptron.inceptron_moe_forward
/// - inceptron.inceptron_moe_forward_shared
/// - inceptron.inceptron_moe_forward_fp8
/// - inceptron.inceptron_moe_forward_fp8_shared
class LowerInceptronOpsPattern : public OpRewritePattern<OperatorOp> {
public:
  using OpRewritePattern<OperatorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(OperatorOp op,
                                 PatternRewriter &rewriter) const override {
    auto opName = op.getName().ltrim("torch.");
    if (!opName.starts_with(kInceptronPrefix))
      return failure();

    // Determine the callee base name (strip "inceptron." prefix).
    std::string calleeBase =
        opName.drop_front(strlen("inceptron.")).str();

    // Filter out !torch.none operands -- they represent optional Tensor? args
    // that were None in the FX graph. The downstream MLIR ops don't have these
    // operands.
    SmallVector<Value> callOperands;
    SmallVector<Type> callOperandTypes;
    for (auto [operand, type] :
         llvm::zip(op.getOperands(), op.getOperandTypes())) {
      if (isa<Torch::NoneType>(type))
        continue;
      callOperands.push_back(operand);
      callOperandTypes.push_back(type);
    }

    // Materialize or retrieve the external function declaration.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto funcType =
        rewriter.getFunctionType(callOperandTypes, op.getResultTypes());
    func::FuncOp callee =
        getOrCreateCallee(module, funcType, calleeBase, op.getLoc(), rewriter);
    if (!callee)
      return failure();

    auto call = rewriter.create<func::CallOp>(op.getLoc(), callee.getSymName(),
                                              op.getResultTypes(), callOperands);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  // Create a mangled function name based on the base name and the function
  // type.
  static std::string mangleFunctionName(const std::string &baseName,
                                         FunctionType funcType) {
    std::string name = baseName;

    auto mangleTensorTypes = [](ValueTensorType type) -> std::string {
      std::string name;
      if (auto sizes = type.getOptionalSizes()) {
        for (auto size : *sizes) {
          if (size == -1) {
            name += "_d";
          } else {
            name += "_" + std::to_string(size);
          }
        }
      } else {
        name += "_u";
      }
      // ingore element type for now
      return name;
    };

    for (Type type : funcType.getInputs()) {
      if (auto tensorType = dyn_cast<ValueTensorType>(type)) {
        name += mangleTensorTypes(tensorType);
      } else if (isa<Torch::IntType>(type)) {
        name += "_i";
      }
      // ingore other non-tensor/non-int types for now
    }

    name += "_ret";

    for (Type type : funcType.getResults()) {
      if (auto tensorType = dyn_cast<ValueTensorType>(type)) {
        name += mangleTensorTypes(tensorType);
      }
    }
    return name;
  }

  func::FuncOp getOrCreateCallee(ModuleOp module, FunctionType funcType,
                                 const std::string &calleeBase, Location loc,
                                 PatternRewriter &rewriter) const {
    SymbolTable symbolTable(module);

    std::string calleeNameWithSuffix =
        mangleFunctionName(calleeBase, funcType);

    if (auto existing =
            symbolTable.lookup<func::FuncOp>(calleeNameWithSuffix)) {
      return existing;
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto callee =
        rewriter.create<func::FuncOp>(loc, calleeNameWithSuffix, funcType);
    callee.setVisibility(SymbolTable::Visibility::Private);
    // Leave the function without a body to mark it as an external
    // declaration.
    return callee;
  }
};

struct LowerInceptronOps : public LowerInceptronOpsBase<LowerInceptronOps> {
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();

    RewritePatternSet patterns(context);
    patterns.add<LowerInceptronOpsPattern>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createLowerInceptronOpsPass() {
  return std::make_unique<LowerInceptronOps>();
}
