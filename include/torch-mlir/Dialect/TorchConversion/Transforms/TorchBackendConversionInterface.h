#ifndef TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_TORCHBACKENDCONVERSIONINTERFACE_H
#define TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_TORCHBACKENDCONVERSIONINTERFACE_H

#include "mlir/IR/DialectInterface.h"

namespace mlir {
class ConversionTarget;
class RewritePatternSet;
class TypeConverter;

namespace torch::TorchConversion {

/// Backend dialects can supply custom operator conversions and extend the
/// linalg-on-tensors contract. Load implementing dialects before running
/// passes, including dialects that do not yet occur in the source IR.
class TorchBackendConversionInterface
    : public DialectInterface::Base<TorchBackendConversionInterface> {
public:
  explicit TorchBackendConversionInterface(Dialect *dialect) : Base(dialect) {}

  /// Supply conversions while Torch lists and scalar constants are still
  /// intact.
  virtual void populateLinalgConversion(TypeConverter &converter,
                                        RewritePatternSet &patterns,
                                        ConversionTarget &target) const = 0;

  virtual void populateTypeConversion(TypeConverter &converter,
                                      RewritePatternSet &patterns,
                                      ConversionTarget &target) const = 0;

  virtual void
  configureLinalgBackendContract(ConversionTarget &target,
                                 const TypeConverter &converter) const = 0;
};

} // namespace torch::TorchConversion
} // namespace mlir

// Resolve identity through the shared MLIR runtime instead of requiring a
// Torch C++ symbol to be exported by the host's Python aggregate library.
MLIR_DECLARE_EXPLICIT_FALLBACK_TYPE_ID(
    mlir::torch::TorchConversion::TorchBackendConversionInterface)

#endif
