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

constexpr StringLiteral kTargetOpName("inceptron.inceptron_scaled_mm");
constexpr StringLiteral kCalleeName("inceptron_scaled_mm");
constexpr StringLiteral kAtenScaledMmOpName("aten._scaled_mm");

/// Rewrites Inceptron custom torch.operators into a direct func.call to an
/// external function declaration. The declaration is created on-demand in the
/// module with a signature that matches the specific operator instance.
class LowerInceptronOpsPattern : public OpRewritePattern<OperatorOp> {
public:
  using OpRewritePattern<OperatorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(OperatorOp op,
                                PatternRewriter &rewriter) const override {
    StringRef name = op.getName().ltrim("torch.");
    if (name != kTargetOpName && name != kAtenScaledMmOpName)
      return failure();

    SmallVector<Type> opdTypes;
    SmallVector<Value> opds;

    if (name == kTargetOpName) {
      Value biasOperand = op.getOperand(5);
      auto constNoneOp = biasOperand.getDefiningOp<ConstantNoneOp>();
      if (!constNoneOp) {
        return rewriter.notifyMatchFailure(op,
                                           "only supports bias=None for now");
      }
      opdTypes.append(op.getOperandTypes().begin(),
                      op.getOperandTypes().begin() + 5);
      opds.append(op.getOperands().begin(), op.getOperands().begin() + 5);
    } else {
      if (!op.getOperand(4).getDefiningOp<ConstantNoneOp>()) {
        return rewriter.notifyMatchFailure(op,
                                           "only supports bias=None for now");
      }
      if (!op.getOperand(5).getDefiningOp<ConstantNoneOp>()) {
        return rewriter.notifyMatchFailure(
            op, "only supports scale_result=None for now");
      }
      auto useFastAccum = op.getOperand(7).getDefiningOp<ConstantBoolOp>();
      if (!useFastAccum || useFastAccum.getValue()) {
        return rewriter.notifyMatchFailure(
            op, "only supports use_fast_accum=False for now");
      }

      opdTypes = {
          op.getOperand(0).getType(),
          op.getOperand(1).getType(),
          op.getOperand(2).getType(),
          op.getOperand(3).getType(),
          op.getOperand(6).getType(),
      };
      opds = {
          op.getOperand(0),
          op.getOperand(1),
          op.getOperand(2),
          op.getOperand(3),
          op.getOperand(6),
      };
    }

    // Materialize or retrieve the external function declaration.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto funcType = rewriter.getFunctionType(opdTypes, op.getResultTypes());
    func::FuncOp callee =
        getOrCreateCallee(module, funcType, op.getLoc(), rewriter);
    if (!callee)
      return failure();

    auto call = rewriter.create<func::CallOp>(op.getLoc(), callee.getSymName(),
                                              op.getResultTypes(), opds);
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
      }
      // ingore non-tensor types for now
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
                                 Location loc,
                                 PatternRewriter &rewriter) const {
    SymbolTable symbolTable(module);

    std::string calleeNameWithSuffix =
        mangleFunctionName(kCalleeName.str(), funcType);

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
