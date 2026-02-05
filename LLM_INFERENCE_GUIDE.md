# LLM Inference: CPU vs GPU System Requirements Guide

## Executive Summary

This guide provides practical guidelines for determining if Large Language Models (LLMs) can run on different hardware configurations, covering CPU-only systems, GPU-accelerated systems, and Apple Silicon's unified memory architecture.

**Key Takeaway**: Memory bandwidth is the primary bottleneck for LLM inference, not raw computational power. Quantization dramatically reduces memory requirements, making CPU-only inference viable for smaller models.

---

## 1. RAM vs VRAM Requirements

### Understanding the Difference

- **RAM (System Memory)**: Used for CPU inference and as overflow when GPU VRAM is insufficient
- **VRAM (GPU Memory)**: Used for GPU-accelerated inference, offers much higher bandwidth than RAM

### Memory Requirements by Model Size

#### Quantized Models (Q4_K_M quantization - recommended)

| Model Size | RAM (CPU) | VRAM (GPU) | Disk Storage |
|------------|-----------|------------|--------------|
| 3B params  | 4-6 GB    | 3-4 GB     | ~2 GB        |
| 7B params  | 8-12 GB   | 6-8 GB     | ~5 GB        |
| 13B params | 16-20 GB  | 10-12 GB   | ~7 GB        |
| 33B params | 32-40 GB  | 20-24 GB   | ~18 GB       |
| 70B params | 64-80 GB  | 40-48 GB   | ~40 GB       |

#### Unquantized Models (FP16/BF16)

| Model Size | RAM (CPU) | VRAM (GPU) | Disk Storage |
|------------|-----------|------------|--------------|
| 7B params  | ~14 GB    | ~14 GB     | ~13 GB       |
| 13B params | ~26 GB    | ~26 GB     | ~25 GB       |
| 33B params | ~66 GB    | ~66 GB     | ~65 GB       |
| 70B params | ~140 GB   | ~140 GB    | ~150 GB      |

**Note**: Quantization reduces memory requirements by 60-75% with minimal quality loss (Q4) or 50% with better quality (Q8).

---

## 2. When Models Can Run on CPU-Only Systems

### CPU-Only Inference is Viable When:

1. **Model size**: 3B-7B parameters with quantization
2. **Use case**: Development, testing, privacy-sensitive applications, offline use
3. **Performance expectations**: 3-8 tokens/second (acceptable for many tasks)

### CPU-Only Inference Limitations:

- **Speed**: 5-20x slower than GPU inference
- **Model size**: Models larger than 7B become extremely slow (often <1 token/sec)
- **Memory bandwidth**: Critical factor - faster RAM = faster inference
- **Thread count**: Only 5-16 threads needed to saturate memory bandwidth; more threads cause overhead

### Recommended CPU-Only Configurations:

- **Minimum**: 8GB RAM, 3B models (Q4 quantization)
- **Comfortable**: 16GB RAM, 7B models (Q4 quantization)
- **Optimal**: 32GB+ RAM, 7B-13B models (Q4-Q5 quantization)

### Tools for CPU Inference:

- **llama.cpp**: Pure C/C++ implementation, excellent portability, AVX/AVX2/AVX512 support
- **Ollama**: User-friendly, automatic quantization, good defaults

---

## 3. Memory Bandwidth Requirements and Impact on Speed

### Why Memory Bandwidth Matters

Memory bandwidth is the **primary bottleneck** for LLM inference, not computational power. During autoregressive decoding, models are bandwidth-bound rather than compute-bound.

### Bandwidth Impact on Performance:

- **CPU**: Performance scales almost proportionally with RAM frequency
  - Dual-channel DDR4-3200: ~50 GB/s
  - Quad-channel DDR5-6000: ~200 GB/s
  - Apple Silicon M1 Max: 400 GB/s unified memory

- **GPU**: High-bandwidth memory (HBM) provides massive advantage
  - Consumer GPUs (GDDR6): 400-600 GB/s
  - Professional GPUs (HBM2/HBM3): 1-3 TB/s

### Real-World Impact:

- **Low bandwidth (50 GB/s)**: ~3-5 tokens/sec on 7B model
- **Medium bandwidth (200 GB/s)**: ~10-15 tokens/sec on 7B model
- **High bandwidth (400+ GB/s)**: ~20-40 tokens/sec on 7B model

### Key Finding:

Even when increasing batch sizes, performance plateaus due to DRAM bandwidth saturation. This is why GPU inference is faster—not just because of compute, but because of superior memory bandwidth.

---

## 4. Apple Silicon Unified Memory Considerations

### Unique Advantages

Apple Silicon's unified memory architecture eliminates the CPU-GPU data transfer bottleneck found in traditional systems. All processors (CPU, GPU, Neural Engine) share the same memory pool without copying.

### Memory Bandwidth by Chip:

| Chip | Unified Memory Bandwidth |
|------|-------------------------|
| M1/M2/M3 (base) | 68-100 GB/s |
| M1/M2/M3 Pro | 200 GB/s |
| M1/M2/M3 Max | 400 GB/s |
| M1/M2/M3 Ultra | 800 GB/s |

### Real-World Performance (MLX Framework):

#### 8GB Unified Memory (Base M1/M2/M3):
- 7B models (Q4): 5-15 tokens/second
- Limited to smaller models or heavy quantization

#### 16GB Unified Memory (M1/M2/M3 Pro):
- 7B models (FP16): 10-25 tokens/second
- 13B models (Q4): Comfortable performance
- Best balance of cost and performance

#### 32GB+ Unified Memory (M1/M2/M3 Max/Ultra):
- 7B models: 15-40 tokens/second
- 13B-30B models: Viable with quantization
- M3 Ultra with 512GB: Can run 100B+ parameter models

### Optimization Tips:

1. **Use MLX framework**: Optimized for Apple Silicon, achieves highest throughput
2. **Quantization is crucial**: 4-bit quantization reduces 7B model from 14GB to ~4GB
3. **Unified memory advantage**: No VRAM limitations—total system memory is available

### Framework Comparison (Apple Silicon):

- **MLX**: Highest sustained throughput, best for Apple Silicon
- **llama.cpp**: Good performance, cross-platform compatibility
- **Ollama**: User-friendly, good defaults, uses llama.cpp under the hood
- **PyTorch MPS**: Lower performance than MLX for LLM inference

---

## 5. Hybrid CPU+GPU Inference (GPU Offloading)

### What is GPU Offloading?

Partial offloading allows models to run with some layers on GPU and some on CPU, extending effective memory capacity beyond VRAM limits.

### When to Use Hybrid Inference:

- **Model doesn't fit entirely in VRAM**: Offload overflow layers to CPU
- **Limited VRAM**: Extend capacity without severe speed impact
- **Cost optimization**: Use available hardware efficiently

### How It Works:

1. **Layer-based offloading**: Specify number of layers to run on GPU (`--n-gpu-layers` in llama.cpp)
2. **Automatic fallback**: When VRAM is full, remaining layers run on CPU
3. **Performance trade-off**: Partial GPU offloading is faster than CPU-only but slower than full GPU

### Performance Characteristics:

- **Full GPU**: Fastest (20-100+ tokens/sec depending on GPU)
- **Partial GPU offload**: Moderate (10-30 tokens/sec)
- **CPU-only**: Slowest (3-8 tokens/sec)

### Implementation (llama.cpp):

```bash
# Offload all possible layers to GPU
./main -m model.gguf --n-gpu-layers 999

# Offload specific number of layers
./main -m model.gguf --n-gpu-layers 32

# Split work between multiple GPUs
./main -m model.gguf --n-gpu-layers 999 --split-mode layer
```

### Supported Backends:

- **NVIDIA**: CUDA + cuBLAS
- **AMD**: HIP
- **Apple Silicon**: Metal (via llama.cpp or MLX)
- **Mobile/Adreno**: OpenCL or Vulkan
- **Other**: SYCL or Vulkan

### Key Insight:

Hybrid inference maintains low computation latency while expanding effective memory capacity. Modern systems can overlap CPU-offloaded tasks with GPU execution during decode phase.

---

## 6. Minimum System Requirements by Model Size

### Consumer Laptops (8-16GB RAM, CPU-only or integrated GPU)

#### 8GB RAM Systems:
- **Viable**: 3B models (Q4 quantization)
- **Performance**: 3-5 tokens/second
- **Use cases**: Code completion, simple Q&A, development/testing
- **Not recommended**: Models larger than 3B

#### 16GB RAM Systems:
- **Viable**: 7B models (Q4 quantization), 3B models (Q8)
- **Performance**: 5-10 tokens/second
- **Use cases**: General-purpose tasks, moderate complexity
- **Not recommended**: Models larger than 7B

**Tools**: Ollama (easiest), llama.cpp (most control)

---

### Gaming PCs (16-32GB RAM + Dedicated GPU)

#### 16GB RAM + 8GB VRAM (e.g., RTX 3060, RTX 4060):
- **Viable**: 7B models (Q4), full GPU acceleration
- **Performance**: 20-40 tokens/second
- **VRAM usage**: ~6-8GB for 7B models
- **Comfortable**: Best balance for consumer hardware

#### 32GB RAM + 12GB VRAM (e.g., RTX 3060 12GB, RTX 4070):
- **Viable**: 13B models (Q4), 7B models (Q8)
- **Performance**: 15-30 tokens/second (13B), 30-50 tokens/second (7B)
- **VRAM usage**: ~10-12GB for 13B models
- **Comfortable**: Excellent for most use cases

#### 32GB RAM + 16GB+ VRAM (e.g., RTX 4080, RTX 4090):
- **Viable**: 13B-22B models (Q4), 7B models (unquantized)
- **Performance**: 20-60 tokens/second depending on model
- **VRAM usage**: Can handle larger models comfortably
- **Optimal**: Best consumer-grade setup

**Tools**: llama.cpp with CUDA, Ollama (automatic GPU detection)

---

### Apple Silicon Macs (M1/M2/M3 with Unified Memory)

#### Base M1/M2/M3 (8GB unified memory):
- **Viable**: 3B models (Q4), 7B models (Q4) with limitations
- **Performance**: 5-15 tokens/second
- **Memory**: Tight constraints, may swap to disk
- **Recommendation**: Upgrade to 16GB if possible

#### M1/M2/M3 Pro (16GB unified memory):
- **Viable**: 7B models (FP16), 13B models (Q4)
- **Performance**: 10-25 tokens/second
- **Memory**: Comfortable for most tasks
- **Sweet spot**: Best price/performance ratio

#### M1/M2/M3 Max (32GB+ unified memory):
- **Viable**: 13B-30B models (Q4), 7B models (unquantized)
- **Performance**: 15-40 tokens/second
- **Memory**: Excellent headroom
- **Optimal**: Professional use cases

#### M1/M2/M3 Ultra (64GB+ unified memory):
- **Viable**: 30B-70B models (Q4), massive models possible
- **Performance**: 20-50+ tokens/second
- **Memory**: Can run largest consumer models
- **Flagship**: Maximum capability

**Tools**: MLX (best performance), Ollama (easiest), llama.cpp (cross-platform)

---

### Workstations (64GB+ RAM, Professional GPUs)

#### 64GB RAM + 24GB VRAM (e.g., RTX 3090, RTX 4090):
- **Viable**: 33B models (Q4), 13B models (unquantized)
- **Performance**: 30-80 tokens/second
- **Use cases**: Professional development, research

#### 64GB+ RAM + 40GB+ VRAM (e.g., A100 40GB):
- **Viable**: 70B models (Q4), 33B models (unquantized)
- **Performance**: 40-100+ tokens/second
- **Use cases**: Production inference, large-scale development

#### 128GB+ RAM + 80GB+ VRAM (e.g., A100 80GB, H100):
- **Viable**: 70B+ models (unquantized), massive models
- **Performance**: 50-200+ tokens/second
- **Use cases**: Enterprise, research, production at scale

**Tools**: llama.cpp with CUDA, vLLM, TensorRT-LLM

---

## Practical Decision Tree

### Step 1: Determine Your Model Size
- **3B**: Consumer laptops, basic tasks
- **7B**: Most common, good balance
- **13B**: Better quality, needs more resources
- **33B+**: Professional/enterprise use

### Step 2: Check Available Memory
- **8GB**: 3B models only (Q4)
- **16GB**: 7B models (Q4), 3B models (Q8)
- **32GB**: 13B models (Q4), 7B models (Q8)
- **64GB+**: 33B+ models possible

### Step 3: Choose Quantization
- **Q4 (4-bit)**: Smallest size, good quality, recommended for most
- **Q5 (5-bit)**: Better quality, slightly larger
- **Q8 (8-bit)**: Best quality, 2x larger than Q4
- **FP16**: Full precision, 4x larger than Q4

### Step 4: Select Framework
- **CPU-only**: llama.cpp, Ollama
- **NVIDIA GPU**: llama.cpp (CUDA), Ollama, vLLM
- **AMD GPU**: llama.cpp (HIP), Ollama
- **Apple Silicon**: MLX (best), Ollama, llama.cpp (Metal)

---

## Performance Expectations Summary

| Hardware | Model Size | Quantization | Tokens/Second |
|----------|------------|--------------|---------------|
| CPU (16GB RAM) | 7B | Q4 | 3-8 |
| CPU (32GB RAM) | 7B | Q4 | 5-12 |
| GPU (8GB VRAM) | 7B | Q4 | 20-40 |
| GPU (16GB VRAM) | 13B | Q4 | 15-30 |
| GPU (24GB VRAM) | 33B | Q4 | 20-50 |
| Apple M1 Pro (16GB) | 7B | Q4 | 10-25 |
| Apple M2 Max (32GB) | 13B | Q4 | 15-35 |
| Apple M3 Ultra (128GB) | 70B | Q4 | 20-50 |

---

## Key Takeaways

1. **Memory bandwidth is king**: Faster memory = faster inference, regardless of CPU/GPU
2. **Quantization is essential**: 4-bit quantization reduces memory by 75% with minimal quality loss
3. **CPU inference is viable**: For 3B-7B models, CPU-only systems work fine for many use cases
4. **Unified memory advantage**: Apple Silicon eliminates CPU-GPU transfer bottlenecks
5. **Hybrid inference extends capacity**: GPU offloading allows larger models on limited VRAM
6. **16GB is the sweet spot**: Comfortable for 7B models, viable for 13B with quantization
7. **Model size vs quality trade-off**: Larger models need more resources but offer better quality

---

## References and Further Reading

- llama.cpp Performance Benchmarks: https://johannesgaessler.github.io/llamacpp_performance
- Ollama VRAM Requirements Guide: https://localllm.in/blog/ollama-vram-requirements-for-local-llms
- Apple MLX Documentation: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- Memory Bandwidth Research: "Efficient LLM Inference: Bandwidth, Compute, Synchronization, and Capacity"
- llama.cpp Quantization Guide: https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/README.md
