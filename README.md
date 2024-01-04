# tfhe_basics
This is to try different tfhe works
Deep learning models made with TensorFlow or Keras should be usable by preliminary converting them to ONNX.
Quantized Aware training ensures that inputs to the linear/Dense and Conv Layers are quantized. 
To use quantized Aware training with torch model,

from concrete.ml.torch.compile import compile_torch_model
n_bits_qat = 3

quantized_module = compile_torch_model(
    torch_model,
    torch_input,
    import_qat=True,
    n_bits=n_bits_qat,
)

QAT can be carried out before or after the training phase. 
QAT allows weights and activations to be reduced to very low bit-widths which when combined with pruning keeps the accumulator bit-widths low. 
Concrete-ML uses Brevitas to perform QAT for Pythorch NNs.
