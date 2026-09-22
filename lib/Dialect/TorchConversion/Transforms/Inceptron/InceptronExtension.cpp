#include "InceptronExtension.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OperationSupport.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::torch;

namespace {
constexpr StringLiteral kInceptronTorchPrefix = "torch.inceptron.inceptron_";
constexpr StringLiteral kScaledMMTorchOp =
    "torch.inceptron.inceptron_scaled_mm";
constexpr StringLiteral kAllReduceTorchOp =
    "torch.inceptron.inceptron_all_reduce";
constexpr StringLiteral kMoeFp8SharedTorchOp =
    "torch.inceptron.inceptron_moe_forward_fp8_shared";
constexpr StringLiteral kMoeW8A16SharedTorchOp =
    "torch.inceptron.inceptron_moe_forward_w8a16_shared";
constexpr StringLiteral kMoeRouteTorchOp =
    "torch.inceptron.inceptron_moe_route";

FailureOr<RankedTensorType>
convertResultType(Torch::OperatorOp op, unsigned index,
                  const TypeConverter &typeConverter) {
  auto type = dyn_cast_or_null<RankedTensorType>(
      typeConverter.convertType(op.getResult(index).getType()));
  if (!type)
    return op.emitOpError()
           << "expected result " << index << " to convert to a ranked tensor";
  return type;
}

Value createDestination(ConversionPatternRewriter &rewriter, Location loc,
                        RankedTensorType type, ValueRange dimensionSources) {
  SmallVector<Value> dynamicDimensions;
  for (auto [dimension, size] : llvm::enumerate(type.getShape())) {
    if (!ShapedType::isDynamic(size))
      continue;
    dynamicDimensions.push_back(rewriter.create<tensor::DimOp>(
        loc, dimensionSources[dimension], dimension));
  }
  return rewriter.create<tensor::EmptyOp>(
      loc, type.getShape(), type.getElementType(), dynamicDimensions);
}

LogicalResult requireRegisteredOperation(Torch::OperatorOp op,
                                         StringRef operationName) {
  if (RegisteredOperationName::lookup(operationName, op.getContext()))
    return success();
  return op.emitOpError()
         << "cannot lower to '" << operationName
         << "' because it is not registered; load the Inceptron dialect "
            "plugin before running the linalg-on-tensors pipeline";
}

class ConvertInceptronOperatorPattern
    : public OpConversionPattern<Torch::OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Torch::OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const StringRef torchName = op.getName();
    if (!torchName.starts_with(kInceptronTorchPrefix))
      return failure();

    if (torchName == kScaledMMTorchOp)
      return lowerScaledMM(op, adaptor, rewriter);
    if (torchName == kAllReduceTorchOp)
      return lowerAllReduce(op, adaptor, rewriter);
    if (torchName == kMoeRouteTorchOp)
      return lowerMoeRoute(op, adaptor, rewriter);
    if (torchName == kMoeFp8SharedTorchOp)
      return lowerMoe(op, adaptor, rewriter, "inceptron.moe_forward_fp8_shared",
                      17);
    if (torchName == kMoeW8A16SharedTorchOp)
      return lowerMoe(op, adaptor, rewriter,
                      "inceptron.moe_forward_w8a16_shared", 20);

    return op.emitOpError()
           << "unsupported Inceptron custom operator '" << torchName << "'";
  }

private:
  LogicalResult lowerScaledMM(Torch::OperatorOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    constexpr StringLiteral operationName = "inceptron.scaled_mm";
    if (failed(requireRegisteredOperation(op, operationName)))
      return failure();
    if (op.getNumOperands() != 6 || op.getNumResults() != 1)
      return op.emitOpError()
             << "expected six operands and one result for scaled MM";
    if (!isa<Torch::NoneType>(op.getOperand(5).getType()))
      return op.emitOpError() << "scaled MM bias is not supported";

    FailureOr<RankedTensorType> resultType =
        convertResultType(op, 0, *getTypeConverter());
    if (failed(resultType))
      return failure();
    ValueRange operands = adaptor.getOperands();
    Value destination = createDestination(rewriter, op.getLoc(), *resultType,
                                          ValueRange{operands[0], operands[1]});
    Value zero = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getZeroAttr(resultType->getElementType()));
    destination =
        rewriter.create<linalg::FillOp>(op.getLoc(), zero, destination)
            .getResult(0);

    OperationState state(op.getLoc(), operationName);
    state.addOperands(operands.take_front(4));
    state.addOperands(destination);
    state.addTypes(*resultType);
    Operation *outDtype = op.getOperand(4).getDefiningOp();
    Operation *bias = op.getOperand(5).getDefiningOp();
    rewriter.replaceOp(op, rewriter.create(state));
    if (outDtype && outDtype->use_empty())
      rewriter.eraseOp(outDtype);
    if (bias && bias->use_empty())
      rewriter.eraseOp(bias);
    return success();
  }

  LogicalResult lowerAllReduce(Torch::OperatorOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
    constexpr StringLiteral operationName = "inceptron.all_reduce";
    if (failed(requireRegisteredOperation(op, operationName)))
      return failure();
    if (op.getNumOperands() != 2 || op.getNumResults() != 1)
      return op.emitOpError()
             << "expected two operands and one result for all-reduce";

    FailureOr<RankedTensorType> resultType =
        convertResultType(op, 0, *getTypeConverter());
    if (failed(resultType))
      return failure();
    ValueRange operands = adaptor.getOperands();
    if (!operands[1].getType().isInteger(64))
      return op.emitOpError() << "all-reduce communicator must lower to i64";
    SmallVector<Value> dimensionSources(resultType->getRank(), operands[0]);
    Value destination =
        createDestination(rewriter, op.getLoc(), *resultType, dimensionSources);

    OperationState state(op.getLoc(), operationName);
    state.addOperands({operands[0], operands[1], destination});
    state.addTypes(*resultType);
    state.addAttribute("operandSegmentSizes",
                       rewriter.getDenseI32ArrayAttr({0, 1, 1, 1}));
    state.addAttribute("resultSegmentSizes",
                       rewriter.getDenseI32ArrayAttr({1, 0}));
    rewriter.replaceOp(op, rewriter.create(state));
    return success();
  }

  LogicalResult lowerMoeRoute(Torch::OperatorOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    constexpr StringLiteral operationName = "inceptron.moe_route";
    if (failed(requireRegisteredOperation(op, operationName)))
      return failure();
    if (op.getNumOperands() != 3 || op.getNumResults() != 2)
      return op.emitOpError()
             << "expected three operands and two results for MoE routing";

    auto topK = op.getOperand(1).getDefiningOp<Torch::ConstantIntOp>();
    if (!topK)
      return op.emitOpError() << "MoE routing top_k must be constant";
    auto softmax = op.getOperand(2).getDefiningOp<Torch::ConstantBoolOp>();
    if (!softmax)
      return op.emitOpError() << "MoE routing softmax flag must be constant";

    FailureOr<RankedTensorType> weightsType =
        convertResultType(op, 0, *getTypeConverter());
    FailureOr<RankedTensorType> idsType =
        convertResultType(op, 1, *getTypeConverter());
    if (failed(weightsType) || failed(idsType))
      return failure();

    Value routerLogits = adaptor.getOperands()[0];
    SmallVector<Value> dimensionSources(idsType->getRank(), routerLogits);
    Value idsDestination =
        createDestination(rewriter, op.getLoc(), *idsType, dimensionSources);
    dimensionSources.resize(weightsType->getRank(), routerLogits);
    Value weightsDestination = createDestination(
        rewriter, op.getLoc(), *weightsType, dimensionSources);

    OperationState state(op.getLoc(), operationName);
    state.addOperands({routerLogits, idsDestination, weightsDestination});
    state.addTypes({*idsType, *weightsType});
    state.addAttribute("top_k", topK.getValueAttr());
    state.addAttribute("softmax", softmax.getValueAttr());
    state.addAttribute("resultSegmentSizes",
                       rewriter.getDenseI32ArrayAttr({2, 0}));
    Operation *route = rewriter.create(state);
    rewriter.replaceOp(op, {route->getResult(1), route->getResult(0)});
    return success();
  }

  LogicalResult lowerMoe(Torch::OperatorOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter,
                         StringRef operationName,
                         unsigned expectedOperandCount) const {
    if (failed(requireRegisteredOperation(op, operationName)))
      return failure();
    if (op.getNumOperands() != expectedOperandCount || op.getNumResults() != 2)
      return op.emitOpError()
             << "unexpected operand or result count for " << operationName;
    if (!isa<Torch::NoneType>(op.getOperand(3).getType()))
      return op.emitOpError()
             << "MoE input_ids must be None for the supported variants";

    FailureOr<RankedTensorType> sharedType =
        convertResultType(op, 0, *getTypeConverter());
    FailureOr<RankedTensorType> fusedType =
        convertResultType(op, 1, *getTypeConverter());
    if (failed(sharedType) || failed(fusedType))
      return failure();

    ValueRange operands = adaptor.getOperands();
    SmallVector<Value> loweredOperands;
    loweredOperands.reserve(operands.size() - 1);
    loweredOperands.append(operands.begin(), operands.begin() + 3);
    loweredOperands.append(operands.begin() + 4, operands.end());

    OperationState state(op.getLoc(), operationName);
    state.addOperands(loweredOperands);
    state.addTypes({*sharedType, *fusedType});
    Operation *inputIds = op.getOperand(3).getDefiningOp();
    rewriter.replaceOp(op, rewriter.create(state));
    if (inputIds && inputIds->use_empty())
      rewriter.eraseOp(inputIds);
    return success();
  }
};

class ConvertTorchConstantIntPattern
    : public OpConversionPattern<Torch::ConstantIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Torch::ConstantIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, rewriter.getIntegerAttr(rewriter.getI64Type(),
                                    op.getValueAttr().getValue()));
    return success();
  }
};

class ConvertTorchConstantBoolPattern
    : public OpConversionPattern<Torch::ConstantBoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Torch::ConstantBoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValueAttr());
    return success();
  }
};

bool isIgnoredInceptronNone(Torch::ConstantNoneOp op) {
  return llvm::all_of(op->getUses(), [](OpOperand &use) {
    auto operatorOp = dyn_cast<Torch::OperatorOp>(use.getOwner());
    if (!operatorOp)
      return false;
    StringRef name = operatorOp.getName();
    unsigned operand = use.getOperandNumber();
    return (name == kScaledMMTorchOp && operand == 5) ||
           ((name == kMoeFp8SharedTorchOp || name == kMoeW8A16SharedTorchOp) &&
            operand == 3);
  });
}
} // namespace

void mlir::torch::TorchConversion::populateInceptronBackendTypeConversion(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  patterns.add<ConvertInceptronOperatorPattern, ConvertTorchConstantIntPattern,
               ConvertTorchConstantBoolPattern>(typeConverter,
                                                patterns.getContext());
  target.addDynamicallyLegalOp<Torch::ConstantNoneOp>(isIgnoredInceptronNone);
}

void mlir::torch::TorchConversion::configureInceptronLinalgBackendContract(
    ConversionTarget &target, TypeConverter &typeConverter) {
  target.addDynamicallyLegalDialect(
      [typeConverter = &typeConverter](Operation *op) {
        return op->getName().isRegistered() && typeConverter->isLegal(op);
      },
      "inceptron");
}
