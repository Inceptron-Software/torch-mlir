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
    if (torchName == "torch.inceptron.inceptron_moe_permute" ||
        torchName == "torch.inceptron.inceptron_moe_w8a8_grouped_mm" ||
        torchName == "torch.inceptron.inceptron_moe_w8a16_grouped_mm" ||
        torchName == "torch.inceptron.inceptron_moe_unpermute")
      return lowerMoeStage(op, adaptor, rewriter);

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

  LogicalResult lowerMoeStage(Torch::OperatorOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    std::string operationName =
        ("inceptron." + op.getName().drop_front(kInceptronTorchPrefix.size()))
            .str();
    if (failed(requireRegisteredOperation(op, operationName)))
      return failure();
    const bool permute = operationName == "inceptron.moe_permute";
    const bool w8a16 = operationName == "inceptron.moe_w8a16_grouped_mm";
    const bool unpermute = operationName == "inceptron.moe_unpermute";
    const unsigned tensorCount = permute ? 3 : (w8a16 || unpermute ? 4 : 5);
    const ValueRange operands = adaptor.getOperands();
    OperationState state(op.getLoc(), operationName);
    state.addOperands(operands.take_front(tensorCount));
    auto addIntAttribute = [&state, &op](unsigned index,
                                         StringRef name) -> LogicalResult {
      auto constant =
          op.getOperand(index).getDefiningOp<Torch::ConstantIntOp>();
      if (!constant)
        return op.emitOpError() << name << " must be constant";
      state.addAttribute(name, constant.getValueAttr());
      return success();
    };
    if (permute && failed(addIntAttribute(4, "row_alignment")))
      return failure();
    if (w8a16 && (failed(addIntAttribute(4, "block_n")) ||
                  failed(addIntAttribute(5, "block_k"))))
      return failure();
    if (permute || unpermute) {
      auto flag =
          op.getOperand(permute ? 5 : 4).getDefiningOp<Torch::ConstantBoolOp>();
      if (!flag)
        return op.emitOpError() << "pre_expert must be constant";
      state.addAttribute("pre_expert", flag.getValueAttr());
    }
    for (unsigned result = 0; result < op.getNumResults(); ++result) {
      auto type = convertResultType(op, result, *getTypeConverter());
      if (failed(type))
        return failure();
      Value destination;
      if (permute && result == 0) {
        auto experts = op.getOperand(3).getDefiningOp<Torch::ConstantIntOp>();
        auto alignment = op.getOperand(4).getDefiningOp<Torch::ConstantIntOp>();
        if (!experts)
          return op.emitOpError() << "num_experts must be constant";
        Value tokens =
            rewriter.create<tensor::DimOp>(op.getLoc(), operands[0], 0);
        Value topK =
            rewriter.create<tensor::DimOp>(op.getLoc(), operands[1], 1);
        Value rows = rewriter.create<arith::MulIOp>(op.getLoc(), tokens, topK);
        Value tail = rewriter.create<arith::ConstantIndexOp>(
            op.getLoc(), experts.getValue() * (alignment.getValue() - 1));
        rows = rewriter.create<arith::AddIOp>(op.getLoc(), rows, tail);
        SmallVector<Value> dynamicDimensions;
        if (type->isDynamicDim(0))
          dynamicDimensions.push_back(rows);
        if (type->isDynamicDim(1))
          dynamicDimensions.push_back(
              rewriter.create<tensor::DimOp>(op.getLoc(), operands[0], 1));
        destination = rewriter.create<tensor::EmptyOp>(
            op.getLoc(), type->getShape(), type->getElementType(),
            dynamicDimensions);
      } else {
        SmallVector<Value> sources;
        if (permute)
          sources.assign(type->getRank(), operands[1]);
        else if (unpermute)
          sources = {operands[2], operands[0]};
        else
          sources = {operands[0], operands[1]};
        destination = createDestination(rewriter, op.getLoc(), *type, sources);
      }
      state.addOperands(destination);
      state.addTypes(*type);
    }
    state.addAttribute("resultSegmentSizes",
                       rewriter.getDenseI32ArrayAttr(
                           {static_cast<int32_t>(op.getNumResults()), 0}));
    rewriter.replaceOp(op, rewriter.create(state));
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
    return name == kScaledMMTorchOp && operand == 5;
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
