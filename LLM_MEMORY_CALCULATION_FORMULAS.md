# LLM Memory Requirements: Mathematical Formulas and Calculations

## Table of Contents
1. [Base Model Memory Calculation](#1-base-model-memory-calculation)
2. [Quantization Levels and Memory Impact](#2-quantization-levels-and-memory-impact)
3. [KV Cache Memory Calculation](#3-kv-cache-memory-calculation)
4. [Activation Memory Requirements](#4-activation-memory-requirements)
5. [Total Inference Memory](#5-total-inference-memory)
6. [Practical Examples](#6-practical-examples)
7. [References](#7-references)

---

## 1. Base Model Memory Calculation

### Core Formula

The base memory required to store model weights is calculated as:

```
Model Weight Memory (GB) = (Parameters × Bytes per Parameter) ÷ 1,073,741,824
```

Where:
- **Parameters** = Total number of model parameters (e.g., 7B, 13B, 70B)
- **Bytes per Parameter** = Depends on quantization/precision format
- **1,073,741,824** = Bytes per GB (1024³)

### Standard Precision Formats

| Format | Bits | Bytes per Parameter | Use Case |
|--------|------|-------------------|----------|
| **FP32** (float32) | 32 | 4.0 | Full precision, training |
| **FP16** (float16) | 16 | 2.0 | Half precision, inference |
| **BF16** (bfloat16) | 16 | 2.0 | Alternative half precision |
| **INT8** (int8) | 8 | 1.0 | 8-bit quantization |
| **INT4** (int4) | 4 | 0.5 | 4-bit quantization |

### Example Calculations

**Llama 2 70B Model:**
- FP32: 70B × 4 bytes = 280 GB
- FP16: 70B × 2 bytes = 140 GB
- INT8: 70B × 1 byte = 70 GB
- INT4: 70B × 0.5 bytes = 35 GB

**Llama 3 8B Model:**
- FP32: 8B × 4 bytes = 32 GB
- FP16: 8B × 2 bytes = 16 GB
- INT8: 8B × 1 byte = 8 GB
- INT4: 8B × 0.5 bytes = 4 GB

---

## 2. Quantization Levels and Memory Impact

### GGUF Quantization Formats (llama.cpp)

GGUF (GPT-Generated Unified Format) is the binary format used by llama.cpp for quantized models. It uses advanced quantization techniques including k-quant methods that optimize different tensor types differently.

#### GGUF Quantization Types

| Format | Bits/Weight | Bytes per Parameter | Size Reduction | Quality |
|--------|------------|-------------------|----------------|---------|
| **Q2_K** | 2.0 | ~0.25 | ~85% | Low |
| **Q3_K_S** | 3.0 | ~0.375 | ~80% | Low-Medium |
| **Q3_K_M** | 3.0 | ~0.375 | ~80% | Medium |
| **Q3_K_L** | 3.0 | ~0.375 | ~80% | Medium-High |
| **Q4_0** | 4.5 | ~0.56 | ~72% | Medium |
| **Q4_1** | 5.0 | ~0.625 | ~69% | Medium |
| **Q4_K_S** | 4.0 | ~0.5 | ~75% | Medium |
| **Q4_K_M** | 4.0 | ~0.5 | ~75% | **Recommended** |
| **Q5_0** | 5.5 | ~0.69 | ~66% | Medium-High |
| **Q5_1** | 6.0 | ~0.75 | ~63% | Medium-High |
| **Q5_K_S** | 5.0 | ~0.625 | ~69% | High |
| **Q5_K_M** | 5.0 | ~0.625 | ~69% | High |
| **Q6_K** | 6.0 | ~0.75 | ~63% | Very High |
| **Q8_0** | 8.5 | ~1.06 | ~47% | Near FP16 |

#### Key Notes on GGUF Formats

- **K-quant variants** (Q4_K_M, Q5_K_M, etc.) use blockwise quantization schemes that minimize reconstruction loss
- **Q4_K_M** is recommended as the best balance for most users (excellent quality with ~75% size reduction)
- **Q5_K_M** or **Q6_K** are preferred when maximum quality is prioritized
- **Q8_0** provides near-FP16 quality with approximately 50% size reduction

#### Example: Llama 3 8B Model File Sizes

| Format | File Size (GB) | vs FP16 |
|--------|---------------|---------|
| FP16 | ~13.0 | Baseline |
| Q8_0 | ~6.7 | ~51% |
| Q5_K_M | ~4.9 | ~62% |
| Q4_K_M | ~3.5 | ~73% |
| Q3_K_M | ~2.6 | ~80% |
| Q2_K | ~1.6 | ~88% |

### Other Quantization Methods

#### Hugging Face Quantization

**LLM.int8()** - 8-bit quantization:
- Uses 1 byte per parameter
- Dynamically preserves higher precision for critical computations
- Maintains inference quality with reduced memory

**QLoRA** - 4-bit quantization:
- Uses 0.5 bytes per parameter
- Includes trainable low-rank adaptation weights
- Enables fine-tuning with minimal memory

---

## 3. KV Cache Memory Calculation

### Core Formula

KV (Key-Value) cache stores attention key and value matrices from previous tokens during inference to avoid redundant computations. This is the primary memory bottleneck for long-context inference.

```
KV Cache Memory (GB) = (2 × Batch_Size × Sequence_Length × Hidden_Dim × Num_Layers × Precision) ÷ 1,073,741,824
```

Where:
- **2** = Factor for both key and value matrices stored per layer
- **Batch_Size** = Number of sequences processed simultaneously
- **Sequence_Length** = Number of tokens in context window
- **Hidden_Dim** = Model's hidden dimension (e.g., 8,192 for Llama 2 70B)
- **Num_Layers** = Number of transformer layers (e.g., 80 for Llama 2 70B)
- **Precision** = Data type size in bytes (FP16 = 2, FP32 = 4, INT8 = 1)

### Alternative Formula (Using Attention Heads)

Some formulations use attention heads instead of hidden dimension:

```
KV Cache Memory (GB) = (2 × Batch_Size × Sequence_Length × Num_Layers × Num_Heads × Head_Dim × Precision) ÷ 1,073,741,824
```

Where:
- **Num_Heads** = Number of attention heads per layer
- **Head_Dim** = Dimensionality of each head's key/value vectors
- Note: `Hidden_Dim = Num_Heads × Head_Dim`

### Key Characteristics

1. **Linear Scaling**: KV cache grows linearly with sequence length (not quadratically), which is why KV caching is essential for efficient inference
2. **Per-Token Storage**: For each token, you store `2 × hidden_dim` elements (keys and values) per layer
3. **Memory Bottleneck**: As context windows expand, KV cache becomes the primary memory bottleneck, often exhausting GPU memory before model weights

### Example Calculations

**Llama 2 70B Model:**
- Batch size: 8
- Sequence length: 4,096 tokens
- Hidden dimension: 8,192
- Number of layers: 80
- Precision: FP16 (2 bytes)

```
KV Cache = (2 × 8 × 4,096 × 8,192 × 80 × 2) ÷ 1,073,741,824
         = 32,768 GB ÷ 1,073,741,824
         ≈ 32 GB
```

**Llama 3 8B Model (1M token context):**
- Batch size: 1
- Sequence length: 1,000,000 tokens
- Hidden dimension: 4,096
- Number of layers: 32
- Precision: FP16 (2 bytes)

```
KV Cache = (2 × 1 × 1,000,000 × 4,096 × 32 × 2) ÷ 1,073,741,824
         = 524,288 GB ÷ 1,073,741,824
         ≈ 128 GB
```

### KV Cache Optimization Techniques

1. **Quantization**: Use FP8 or INT8 for KV cache (reduces memory by 2-4×)
2. **Token Pruning**: Remove less important tokens from cache
3. **Memory Offloading**: Move KV cache to CPU/disk when not actively used
4. **Sliding Window**: Maintain only recent tokens in cache

---

## 4. Activation Memory Requirements

### Definition

Activation memory stores intermediate tensors computed during the forward pass. Unlike training, inference requires minimal activation memory since activations don't need to be retained for backpropagation.

### Formula

For transformer inference, activation memory scales approximately linearly with sequence length:

```
Activation Memory (GB) ≈ (Batch_Size × Sequence_Length × Hidden_Dim × Num_Layers × 2) ÷ 1,073,741,824
```

Where the factor of 2 accounts for intermediate computations in attention and MLP layers.

### Simplified Formula

A more practical approximation:

```
Activation Memory ≈ Max_Tokens × Constant_Factor
```

Where `Constant_Factor` depends on the model architecture and is typically pre-computed by inference engines like TensorRT-LLM.

### Key Characteristics

1. **Linear Scaling**: Activation memory scales linearly with batch size and sequence length
2. **Model-Dependent**: The constant factor varies by model architecture (number of layers, hidden dimension, MLP size)
3. **Inference vs Training**: Training requires much more activation memory (must save for backpropagation), while inference can discard activations immediately

### Example Calculations

**Llama 2 70B Model:**
- Batch size: 1
- Sequence length: 2,048 tokens
- Hidden dimension: 8,192
- Number of layers: 80

```
Activation Memory ≈ (1 × 2,048 × 8,192 × 80 × 2) ÷ 1,073,741,824
                 ≈ 2.5 GB
```

**With batch size 8:**
```
Activation Memory ≈ (8 × 2,048 × 8,192 × 80 × 2) ÷ 1,073,741,824
                 ≈ 20 GB
```

**Llama 3 8B Model (1M token context):**
- Batch size: 1
- Sequence length: 1,000,000 tokens
- Hidden dimension: 4,096
- Number of layers: 32

```
Activation Memory ≈ (1 × 1,000,000 × 4,096 × 32 × 2) ÷ 1,073,741,824
                 ≈ 64 GB
```

### Optimization Techniques

1. **Kernel Fusion**: Combine operations to reduce intermediate tensor storage
2. **Flash Attention**: Reduces activation memory through recomputation
3. **Gradient Checkpointing**: Not applicable to inference (training only)
4. **Sequence Parallelism**: Distribute activations across devices

---

## 5. Total Inference Memory

### Complete Formula

Total GPU memory required for LLM inference is the sum of all components:

```
Total Memory (GB) = Model_Weights + KV_Cache + Activation_Memory + Overhead
```

Where:
- **Model_Weights** = Static model parameters (from Section 1)
- **KV_Cache** = Key-value cache (from Section 3)
- **Activation_Memory** = Intermediate tensors (from Section 4)
- **Overhead** = System overhead, CUDA context, etc. (typically 1-2 GB)

### Practical Example: Llama 3 8B with 1M Token Context

| Component | Memory (GB) | Notes |
|-----------|-------------|-------|
| Model Weights (Q4_K_M) | 3.5 | Quantized weights |
| KV Cache | 128 | FP16, 1M tokens |
| Activation Memory | 64 | Forward pass buffers |
| System Overhead | 2 | CUDA, OS, etc. |
| **Total** | **197.5 GB** | |

### Memory Breakdown Insights

1. **Short Context (< 4K tokens)**: Model weights dominate
2. **Medium Context (4K-32K tokens)**: KV cache becomes significant
3. **Long Context (> 32K tokens)**: KV cache dominates, often 2-4× model size
4. **Very Long Context (> 100K tokens)**: KV cache can be 10×+ model size

### Example: Llama 2 70B Inference

**Scenario 1: Short Context (2K tokens, batch=1)**
- Model Weights (FP16): 140 GB
- KV Cache: ~16 GB
- Activation Memory: ~2.5 GB
- Overhead: 2 GB
- **Total: ~160.5 GB**

**Scenario 2: Long Context (32K tokens, batch=8)**
- Model Weights (FP16): 140 GB
- KV Cache: ~256 GB
- Activation Memory: ~40 GB
- Overhead: 2 GB
- **Total: ~438 GB**

---

## 6. Practical Examples

### Example 1: Llama 3 8B on Consumer GPU

**Goal**: Run Llama 3 8B with 4K context on 24GB GPU

**Solution**:
- Use Q4_K_M quantization: 3.5 GB (model weights)
- KV Cache (4K tokens): ~1 GB
- Activation Memory: ~0.25 GB
- Overhead: 1 GB
- **Total: ~5.75 GB** ✅ Fits comfortably

### Example 2: Llama 2 70B on A100 (80GB)

**Goal**: Run Llama 2 70B with 8K context, batch=4

**Solution**:
- Use FP16: 140 GB (model weights) ❌ Doesn't fit
- Use INT8: 70 GB (model weights)
- KV Cache (8K tokens, batch=4): ~64 GB
- Activation Memory: ~10 GB
- Overhead: 2 GB
- **Total: ~146 GB** ❌ Still doesn't fit

**Better Solution**:
- Use Q4_K_M: ~35 GB (model weights)
- KV Cache (FP16): ~64 GB
- Activation Memory: ~10 GB
- Overhead: 2 GB
- **Total: ~111 GB** ❌ Still exceeds 80GB

**Optimal Solution**:
- Use Q4_K_M: ~35 GB
- KV Cache (INT8): ~32 GB
- Activation Memory: ~10 GB
- Overhead: 2 GB
- **Total: ~79 GB** ✅ Fits with margin

### Example 3: Long Context Inference (1M tokens)

**Model**: Llama 3 8B
**Context**: 1,000,000 tokens

**Memory Breakdown**:
- Model Weights (Q4_K_M): 3.5 GB
- KV Cache (FP16): 128 GB ⚠️ Dominates
- Activation Memory: 64 GB
- Overhead: 2 GB
- **Total: 197.5 GB**

**Optimization**: Use INT8 KV cache
- KV Cache (INT8): 64 GB
- **New Total: 133.5 GB** (33% reduction)

---

## 7. References

### Official Documentation

1. **llama.cpp Quantization Guide**
   - GitHub: https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/README.md
   - Quantization types and bytes per parameter specifications

2. **Hugging Face Accelerate**
   - Documentation: https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator
   - Model memory estimation tool and quantization support

3. **TensorRT-LLM Memory Usage**
   - Documentation: https://nvidia.github.io/TensorRT-LLM/reference/memory.html
   - Activation memory and inference optimization

### Technical Articles

4. **Understanding LLM Memory Requirements**
   - Vikarna's Substack: https://vikarna.substack.com/p/understanding-llm-memory-requirements
   - Comprehensive memory calculation formulas

5. **KV Cache Calculation**
   - Rohan's Bytes: https://rohan-paul.com/p/how-to-calculate-size-of-kv-cache
   - Detailed KV cache formulas and examples

6. **Estimating LLM Inference Memory**
   - TensorWave Blog: https://tensorwave.com/blog/estimating-llm-inference-memory-requirements
   - Practical memory estimation guide

### Academic Papers

7. **Scaling Transformer Inference**
   - Paper on transformer inference optimization and memory requirements

8. **Reducing Activation Recomputation**
   - arXiv:2205.05198
   - Activation memory optimization techniques

### Community Resources

9. **GGUF Format Guide**
   - Medium: Comprehensive Analysis of GGUF Variants
   - Detailed quantization format comparisons

10. **KV Cache Calculator**
    - Online Tool: https://lmcache.ai/kv_cache_calculator.html
    - Interactive KV cache memory calculator

---

## Quick Reference: Memory Calculation Cheat Sheet

### Model Weights
```
Memory (GB) = Parameters × Bytes_per_Param ÷ 1,073,741,824
```

### KV Cache
```
KV Cache (GB) = (2 × B × S × H × L × P) ÷ 1,073,741,824
```
Where: B=batch, S=sequence, H=hidden_dim, L=layers, P=precision_bytes

### Activation Memory
```
Activation (GB) ≈ (B × S × H × L × 2) ÷ 1,073,741,824
```

### Bytes per Parameter Reference
- FP32: 4.0 bytes
- FP16: 2.0 bytes
- INT8: 1.0 bytes
- INT4: 0.5 bytes
- Q4_K_M: ~0.5 bytes
- Q5_K_M: ~0.625 bytes
- Q8_0: ~1.06 bytes

---

*Document compiled from authoritative sources including llama.cpp, Hugging Face, TensorRT-LLM documentation, and academic papers on transformer inference.*
