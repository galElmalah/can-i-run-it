# KV Cache Memory Requirements for LLM Inference

## Executive Summary

KV (Key-Value) cache memory grows **linearly with context length**, making it a critical bottleneck for LLM inference. Unlike model weights (which are fixed), KV cache memory consumption increases with each token in the context, directly impacting GPU memory requirements and batch size capacity.

---

## 1. KV Cache Memory Formula

### Basic Formula

The KV cache memory required per token follows this formula:

```
Memory per token (bytes) = 2 × num_layers × num_heads × head_dim × bytes_per_element
```

Or equivalently:

```
Memory per token (bytes) = 2 × num_layers × hidden_size × bytes_per_element
```

Where:
- **2**: Accounts for both Key (K) and Value (V) vectors
- **num_layers**: Number of transformer decoder layers
- **num_heads**: Number of attention heads (or num_kv_heads for GQA models)
- **head_dim**: Dimension of each attention head (typically `hidden_size / num_heads`)
- **hidden_size**: Total hidden dimension (d_model)
- **bytes_per_element**: 
  - FP32: 4 bytes
  - FP16/BF16: 2 bytes (most common)
  - INT8: 1 byte
  - FP8: 1 byte (emerging)

### Total KV Cache Memory

For a complete sequence:

```
Total KV Cache (bytes) = Memory per token × Sequence Length × Batch Size
```

### Simplified Formula

For models using Grouped Query Attention (GQA) or Multi-Query Attention (MQA):

```
Memory per token = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
```

Where `num_kv_heads` is typically less than `num_heads` (e.g., 8 KV heads for 64 query heads).

---

## 2. Relationship Between Model Dimensions and KV Cache

### Key Relationships

1. **Linear scaling with layers**: Each layer stores its own KV cache
2. **Linear scaling with hidden size**: Larger models require more memory per token
3. **Linear scaling with sequence length**: Longer contexts require proportionally more memory
4. **Linear scaling with batch size**: Each request in a batch needs its own KV cache

### Model Architecture Impact

The KV cache size is determined by:
- **Model depth** (num_layers): More layers = more cache per token
- **Model width** (hidden_size): Larger hidden dimensions = more cache per token
- **Attention mechanism**: 
  - Standard multi-head attention: Uses all heads
  - GQA/MQA: Uses fewer KV heads, reducing cache size

### Memory vs. Compute Trade-off

- **Without KV cache**: Quadratic compute complexity (recompute attention for all tokens)
- **With KV cache**: Linear memory complexity, constant compute per token

---

## 3. Memory Per Token Calculation

### Standard Calculation (FP16/BF16)

For a transformer with:
- `L` layers
- `H` heads
- `D` hidden dimension
- `d_head = D / H` (head dimension)

```
Memory per token = 2 × L × H × d_head × 2 bytes
                 = 2 × L × D × 2 bytes
                 = 4 × L × D bytes
```

### Example Calculations

**LLaMA 3-70B:**
- Layers: 80
- Hidden size: 8,192
- KV heads: 8 (GQA)
- Head dimension: 128
- Precision: FP16 (2 bytes)

```
Memory per token = 2 × 80 × 8 × 128 × 2 = 160 KB per token
```

**For 32K context:**
```
Total KV cache = 160 KB × 32,768 = 5.3 GB
```

**For batch size 32, sequence length 8K:**
```
Total KV cache = 160 KB × 8,192 × 32 = 41.9 GB
```

**LLaMA 2 70B (standard multi-head):**
- Layers: 80
- Hidden size: 8,192
- Heads: 64
- Head dimension: 128
- Precision: FP16

```
Memory per token = 2 × 80 × 64 × 128 × 2 = 2.56 MB per token
```

**For 4K context, batch size 8:**
```
Total KV cache = 2.56 MB × 4,096 × 8 = 83.9 GB
```

**52B Parameter Model (Anthropic-style):**
- Layers: 64
- Hidden size: 8,192
- Heads: 64
- Precision: FP16

```
Memory per token = 2 × 64 × 8,192 × 2 = 2.0 MB per token
```

**For 8K context:**
```
Total KV cache = 2.0 MB × 8,192 = 16.4 GB
```

---

## 4. Common Model Specifications

### LLaMA Model Family

| Model | Parameters | Layers | Hidden Size | Heads | KV Heads | Memory/Token (FP16) |
|-------|-----------|--------|-------------|-------|----------|---------------------|
| LLaMA 3-8B | 8B | 32 | 4,096 | 32 | 8 | 32 KB |
| LLaMA 3-70B | 70B | 80 | 8,192 | 64 | 8 | 160 KB |
| LLaMA 2-7B | 7B | 32 | 4,096 | 32 | 32 | 64 KB |
| LLaMA 2-13B | 13B | 40 | 5,120 | 40 | 40 | 80 KB |
| LLaMA 2-70B | 70B | 80 | 8,192 | 64 | 64 | 2.56 MB |

### Practical Examples

**LLaMA 3-70B on 4× A100 (40GB each):**
- Model weights (INT8): ~70 GB
- Available for KV cache: ~90 GB
- Max tokens per batch: 90 GB / 160 KB = ~562,500 tokens
- At batch size 32: ~17,500 tokens per request
- At batch size 1: ~562,500 tokens per request

**LLaMA 2-70B on 4× A100 (40GB each):**
- Model weights (FP16): ~140 GB (requires 4 GPUs)
- Available for KV cache: ~20 GB
- Max tokens per batch: 20 GB / 2.56 MB = ~7,800 tokens
- At batch size 8: ~975 tokens per request
- At batch size 1: ~7,800 tokens per request

---

## 5. Long Context Handling Mechanisms

### Standard Attention Limitations

Standard transformer attention has **O(n²)** complexity, making long contexts computationally expensive. KV cache reduces this to **O(n)** memory but still grows linearly.

### Efficient Attention Mechanisms

#### 1. **Sliding Window Attention (SWA)**
- Restricts attention to a fixed-size local window
- Reduces complexity from O(n²) to O(n)
- **Challenge**: Training-inference mismatch when adapting full-attention models
- **Solutions**: 
  - SWAA (Sliding Window Attention Adaptation): Plug-and-play adaptation
  - Preserve "sink" tokens (initial tokens)
  - Interleave full-attention and SWA layers
  - Achieves 30-100% speedups for long contexts

#### 2. **Grouped Query Attention (GQA)**
- Reduces number of KV heads (e.g., 8 KV heads for 64 query heads)
- Reduces KV cache size by factor of `num_heads / num_kv_heads`
- Used in LLaMA 3, Mistral, and other modern models

#### 3. **Multi-Query Attention (MQA)**
- Single KV head shared across all query heads
- Maximum KV cache reduction
- May impact quality for some tasks

#### 4. **Token Compression/Eviction**
- **Scissorhands**: Retains important tokens
- **StreamingLLM**: Maintains sliding window + sink tokens
- **H2O**: Head-wise token selection
- **SnapKV**: Token pruning
- **MorphKV**: Constant-sized cache with correlation-aware selection (52.9% memory savings)
- **XKV**: Personalized layer allocation (61.6% average reduction)

#### 5. **Dynamic Memory Compression (DMC)**
- Reduces KV cache overhead by 4-7×
- Maintains acceptable performance

#### 6. **Data Type Optimization**
- **FP8**: Emerging format for KV cache compression
- Reduces memory by 2× compared to FP16
- Maintains reasonable quality

---

## 6. Practical Context Length Limits

### Memory Constraints

The practical context length limit depends on:

1. **Available GPU Memory**: Total VRAM minus model weights
2. **Batch Size**: Larger batches reduce max context per request
3. **Model Architecture**: GQA models support longer contexts
4. **Precision**: Lower precision (INT8, FP8) allows longer contexts

### Rule of Thumb Calculations

**For a model with KV cache size `M` bytes per token:**

```
Max context length = (Available GPU Memory - Model Weights) / (M × Batch Size)
```

**Example: LLaMA 3-70B on single A100 (40GB)**
- Model weights (INT8): ~70 GB (doesn't fit, need multiple GPUs)
- On 4× A100: ~90 GB available for KV cache
- Memory per token: 160 KB
- Batch size 1: 90 GB / 160 KB = **562,500 tokens**
- Batch size 32: 90 GB / (160 KB × 32) = **17,500 tokens per request**

### Performance Degradation

Research shows that:
- **Quality degradation** accelerates as context expands
- Models claiming 100K+ token support often show degradation
- **Extractive compression** outperforms other methods (up to 10× compression)
- Performance varies significantly between dense transformers and MoE architectures

### Practical Recommendations

1. **For 7B-13B models:**
   - Single GPU (24GB): ~4K-8K tokens max context
   - Single GPU (40GB): ~8K-16K tokens max context
   - Batch size 1-4 typically

2. **For 70B models:**
   - 4× A100 (40GB): ~17K-20K tokens per request (batch size 32)
   - 4× A100 (40GB): ~500K+ tokens (batch size 1)
   - Requires model parallelism

3. **Optimization strategies:**
   - Use GQA/MQA models for longer contexts
   - Apply token compression/eviction for very long contexts
   - Use lower precision (INT8, FP8) for KV cache
   - Consider sliding window attention for specific use cases

---

## 7. Memory Breakdown Example

### Complete Memory Requirements

For inference, total GPU memory includes:

1. **Model Weights:**
   - FP32: 4 bytes per parameter
   - FP16/BF16: 2 bytes per parameter
   - INT8: 1 byte per parameter
   - INT4: 0.5 bytes per parameter

2. **KV Cache:**
   - Calculated using formulas above
   - Grows with context length and batch size

3. **Activation Memory:**
   - Intermediate activations during forward pass
   - Typically small compared to weights and KV cache
   - ~5-10% overhead for large models

### Example: LLaMA 3-70B Inference

**Configuration:**
- Batch size: 32
- Sequence length: 8,192 tokens
- Precision: INT8 for weights, FP16 for KV cache

**Memory breakdown:**
- Model weights (INT8): 70 GB
- KV cache: 160 KB × 8,192 × 32 = 41.9 GB
- Activations: ~5 GB (estimated)
- **Total: ~117 GB**

**On 4× A100 (40GB = 160GB total):**
- Utilization: 73%
- Remaining: ~43 GB for additional batch capacity or longer contexts

---

## 8. Key Takeaways

1. **KV cache grows linearly** with context length, making it the dominant memory consumer for long contexts

2. **Memory per token** depends on model architecture:
   - Standard attention: `2 × layers × hidden_size × bytes`
   - GQA: `2 × layers × kv_heads × head_dim × bytes`

3. **Practical limits** depend on:
   - Available GPU memory
   - Batch size requirements
   - Model architecture (GQA helps significantly)

4. **Optimization strategies**:
   - Use GQA/MQA architectures
   - Apply token compression/eviction
   - Use lower precision formats
   - Consider sliding window attention

5. **For common model sizes:**
   - **7B models**: ~4K-8K tokens on single GPU
   - **13B models**: ~8K-16K tokens on single GPU
   - **70B models**: ~17K-20K tokens per request (batch 32) on 4× A100

---

## References

1. [Transformer Inference Arithmetic - kipply's blog](https://kipp.ly/blog/transformer-inference-arithmetic/)
2. [KV Cache Size Calculator](https://lmcache.ai/kv_cache_calculator.html)
3. [Dialogue Without Limits: Constant-Sized KV Caches](https://arxiv.org/html/2503.00979v1)
4. [Sliding Window Attention Adaptation (SWAA)](https://arxiv.org/abs/2502.18845)
5. [Characterizing Prompt Compression Methods for Long Context Inference](https://arxiv.org/html/2407.08892v1)
6. [Dynamic Memory Compression: Retrofitting LLMs](https://arxiv.org/html/2403.09636v2)

---

## Appendix: Quick Reference Formulas

### KV Cache Memory Per Token
```
M_token = 2 × L × H_kv × d_head × bytes_per_element
```

### Total KV Cache Memory
```
M_total = M_token × seq_len × batch_size
```

### Max Context Length
```
max_context = (GPU_memory - model_weights) / (M_token × batch_size)
```

### Memory Bandwidth Bound Threshold
```
min_batch_size = GPU_memory_bandwidth / GPU_flops
```
(For A100: ~208 tokens)
