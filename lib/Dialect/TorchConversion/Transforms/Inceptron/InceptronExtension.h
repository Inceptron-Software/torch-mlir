#ifndef TORCHMLIR_LIB_DIALECT_TORCHCONVERSION_TRANSFORMS_INCEPTRON_INCEPTRONEXTENSION_H
#define TORCHMLIR_LIB_DIALECT_TORCHCONVERSION_TRANSFORMS_INCEPTRON_INCEPTRONEXTENSION_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::torch::TorchConversion {

void populateInceptronBackendTypeConversion(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            ConversionTarget &target);

void configureInceptronLinalgBackendContract(ConversionTarget &target,
                                             TypeConverter &typeConverter);

} // namespace mlir::torch::TorchConversion

#endif // TORCHMLIR_LIB_DIALECT_TORCHCONVERSION_TRANSFORMS_INCEPTRON_INCEPTRONEXTENSION_H
