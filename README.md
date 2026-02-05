# Can I Run It? (Ollama Edition)

A loud **neo‑brutalist** web app that helps you figure out which **Ollama models** your machine can run locally.

- **Search 50+ models** from the Ollama library
- **See instant verdicts** (Can run / Maybe / Too big) based on your RAM
- **Click any model** to see all variants with detailed memory breakdowns
- **Test your local Ollama** installation right from the browser

## Features

### Model Search
Search and filter models by name, category (General, Code, Vision, Reasoning, etc.), or size. Each card shows whether the default variant can run on your machine.

### 📚 Comprehensive System Requirements Guide
See [`LLM_INFERENCE_GUIDE.md`](./LLM_INFERENCE_GUIDE.md) for detailed research on:
- RAM vs VRAM requirements for different model sizes
- CPU-only inference capabilities and limitations
- Memory bandwidth impact on performance
- Apple Silicon unified memory considerations
- Hybrid CPU+GPU inference (GPU offloading)
- Minimum system requirements for consumer laptops, gaming PCs, Apple Silicon Macs, and workstations

### Detailed Variant View
Click a model to see all available variants (different sizes and quantizations). Each variant shows:
- Memory estimate (weights + KV cache + overhead)
- Your available RAM vs. required RAM
- Copy-able `ollama run` command

### System Detection
The app detects what your browser can see:
- CPU threads (`navigator.hardwareConcurrency`)
- RAM (`navigator.deviceMemory` — Chromium only)
- GPU (WebGL renderer string)
- Manual RAM override for browsers that don't expose it

### Local Ollama Test
Optional panel that pings your local Ollama (`localhost:11434`) to check if it's running and list installed models.

## Local dev

```bash
npm install
npm run dev
```

Open the dev URL Vite prints (usually `http://localhost:5173`).

## Build / preview

```bash
npm run build
npm run preview
```

Output goes to `dist/`.

## Deploy to Cloudflare Pages

This is a static SPA:
- **Build command**: `npm run build`
- **Output directory**: `dist`

### Cloudflare Functions (optional)

There's a stub at `functions/api/refresh-models.ts` for a future model refresh endpoint. To implement:
1. Add a D1 database or KV namespace in Cloudflare
2. Implement scraping of `https://ollama.com/library`
3. Set up a cron trigger for daily refresh

## Model Database

Models are stored in `src/data/models.ts` as a static TypeScript file. This includes:
- ~50 popular Ollama models
- Variants with parameter counts and download sizes
- Categories and tags for search

To update: edit the file directly or implement the Cloudflare Worker refresh.

## Ollama CORS Notes

The "Test my Ollama" panel calls `http://localhost:11434`. This may be blocked by:
- **CORS**: Set `OLLAMA_ORIGINS` to include your origin:
  ```bash
  export OLLAMA_ORIGINS="http://localhost:5173,https://your-domain.pages.dev"
  ollama serve
  ```
- **HTTPS → HTTP**: Browsers may block requests from HTTPS pages to `http://localhost`. Run the app locally (`npm run dev`) for full Ollama testing.

## Tech Stack

- **Vite + React + TypeScript**
- **Neo-brutalist CSS** (no framework, just CSS variables)
- **Bungee + IBM Plex Sans** fonts
- **Static build** (Cloudflare Pages ready)

## License

MIT
