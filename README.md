# llama.cpp Metal for Apple GPU family 5, with LFM3 speed-ups

This repository contains a patch for llama.cpp. The patch lets llama.cpp run inference on the GPU of Apple GPU family 5 devices (A12, A12X, A12Z). It also adds speed-ups for the LiquidAI LFM3 model on these devices: an output sketch and a speculative-decoding draft head.

Without the patch, llama.cpp runs the matrix multiplications, the norms and the softmax on the CPU of these devices.

The iPad app that uses this patch is in a separate repository: [lfm3-ipad](https://github.com/0xbyt4/lfm3-ipad).

## The problem

Upstream llama.cpp Metal kernels use SIMD-group reductions and SIMD-group matrix operations. Apple GPU family 5 has neither feature. Upstream llama.cpp detects this and refuses MUL_MAT, NORM, RMS_NORM, SOFT_MAX and SSM_CONV on such a GPU. The scheduler then puts these ops on the CPU.

The patch adds compat kernels for these ops. The compat kernels do reductions through threadgroup memory. They need no SIMD-group instructions.

## Devices

Apple GPU family 5 is the GPU of the A12 chip family:

- A12 Bionic: iPhone XS, iPhone XS Max, iPhone XR, iPad mini (5th generation), iPad Air (3rd generation), iPad (8th generation).
- A12X Bionic: iPad Pro 11-inch (1st generation), iPad Pro 12.9-inch (3rd generation).
- A12Z Bionic: iPad Pro 11-inch (2nd generation), iPad Pro 12.9-inch (4th generation).

The author tested only the A12Z (iPad Pro 11-inch, 2nd generation). The author did not test the other devices.

The patch selects the compat kernels for every GPU without SIMD-group reduction. Upstream llama.cpp sets SIMD-group reduction only for family Apple7 (A14) and later, and for GPUs with Metal 3 support. The A13 (family 6) and older GPUs, for example the A11 (family 4), therefore also select the compat path. The author did not test these GPUs.

## Contents

| File | Purpose |
|---|---|
| `apple5-lfm3.patch` | The patch for llama.cpp commit `338fce6511c744893cc0d3d958b275f40552c7a4` (2026-10-27). |
| `scripts/setup.sh` | Gets llama.cpp at the base commit and applies the patch. |
| `scripts/add_output_sketch.py` | Copies an LFM3 GGUF file and adds the output sketch. |
| `scripts/add_draft_head.py` | Copies an LFM3 GGUF file with the output sketch and adds a draft head from a safetensors file. |
| `LICENSE` | MIT license for the files in this repository. |
| `LICENSE-llama.cpp` | The llama.cpp license. The patch changes llama.cpp files. The changed files keep the llama.cpp license. |

## What the patch changes

### Compat kernels: `ggml/src/ggml-metal/kernels/compat.metal` (new file)

- A tiled matrix-matrix kernel for prompts. The tile is 64 x 64. A threadgroup has 256 threads. Tiles are half precision. Accumulation is float.
- A matrix-vector kernel for generation.
- Q4_0 kernels for 1 to 8 src1 columns. Generation uses 1 column. A speculative-decoding verify step uses more than 1 column. A thread works on several rows and uses each src1 load for all of its rows. src1 and the dot products stay float.
- A Q1_0 kernel for 1 to 8 src1 columns. The LFM3 output sketch uses it.
- An F16 matrix-vector kernel for attention with grouped-query heads. It reads each K row once for all query heads of the group.
- NORM, RMS_NORM and SOFT_MAX kernels for F32 input. NORM and RMS_NORM support the fused MUL and ADD.
- Supported weight types: f32, f16, q1_0, q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K, q5_K, q6_K, iq4_nl, iq4_xs.

### Fixes and speed-ups in existing Metal files

- SSM_CONV runs on GPUs without SIMD-group reduction. Its kernels use no SIMD-group operations. The short-conv layers of LFM3 use this op.
- Half scales: the A12 GPU flushes a subnormal half to zero. Some block scales are subnormal, for example in Q6_K token embeddings. The patch converts the scale bits to float with integer operations, so these scales are not lost.
- Norm fusion: the patch fuses RMS_NORM + MUL + ADD + RMS_NORM + MUL into one kernel. This is a post-norm, the residual add and the next pre-norm.
- Top-k: a prefilter for the radix-select top-k kernel. The kernel first keeps only the values at or above a lower bound, up to 2048 values. Then it selects on this short list. If more values pass, it selects on the full row. The output sketch uses it.
- SOFT_MAX on the compat path uses more threads for few long rows. This makes attention faster at long contexts.
- Residency keep-alive changes, for apps that go to the background:
  - Upstream wakes the keep-alive thread every 5 ms. With the patch, the thread stops when the keep-alive time after the last graph computation ends. It waits for the next graph computation.
  - The new function `ggml_backend_metal_residency_idle()` stops the keep-alive until the next graph computation. Call it when an iOS app goes to the background. A background iOS app must not submit GPU work.

### LFM3 changes

- **Output sketch.** The LFM3 output matrix has 128000 rows (278 MB in Q8_0). The sketch is a Q1_0 copy of this matrix (37 MB): sign bits and one scale for each 128 weights. When the file has `output_sketch.weight` and a decode has 1 to 8 output rows, the graph does these steps:
  1. Rank the vocabulary with the sketch.
  2. Compute exact logits for the 512 best tokens and for the first 1024 token ids.
  3. Set all other logits to -inf.

  The output is not bit-identical to the model without the sketch. In the author's benchmark on 2026-10-29, the first 128 greedy tokens of the essay prompt were the same with and without the sketch.
- **Draft head.** The draft head needs the output sketch. When the file has the `draft_*` tensors and the draft head is on, each decode with 1 to 8 output rows also predicts a draft token. The draft token is the token after the greedy token. A verify step then decodes the sampled token and the draft token together. The draft head is one SwiGLU block. It runs in the same graph as the model.
- New C API in `include/llama.h` (experimental):
  - `llama_set_draft_head(ctx, true)` turns the draft head on.
  - `llama_get_draft_ith(ctx, i)` returns `{ draft token, greedy next token }` as floats for output row `i`. The draft is valid only when the sampled token is the greedy token.
  - In a two-token verify step, both output rows hold the same draft. Request logits for all tokens of the verify step.
  - Create the context with `n_rs_seq` of 1 or more. Then remove a rejected draft with `llama_memory_seq_rm`.

### Environment variables for tests and measurements

| Variable | Effect |
|---|---|
| `GGML_METAL_SIMULATE_APPLE5=1` | Turns off the SIMD-group features on a newer GPU. Use it to test the compat path on a Mac. |
| `GGML_METAL_PROFILE=1` | Runs each op of a single-token graph in its own command buffer. Logs the GPU time per op type when the context is freed. On the A12Z, the GPU clock drops between command buffers. Compare only the proportions of the op times. |
| `GGML_METAL_SKIP_OPS=...` | Does not encode the named ops. The output is wrong. The change in token time shows the cost of the skipped ops. |
| `GGML_METAL_DUP_OPS=...` | Encodes the named ops two times. The change in token time shows the cost of one more such op. Ops that change their input in place give wrong results. |
| `GGML_METAL_N_CB=2` | Sets the number of extra command-buffer encoding threads. The default is 1. |

The op list of `GGML_METAL_SKIP_OPS` and `GGML_METAL_DUP_OPS` is comma-separated. An entry has one of these forms:

- An op name, for example `RMS_NORM`.
- `MUL_MAT:<weight type>`, for example `MUL_MAT:q4_0`.
- `MUL_MAT@<weight tensor name>`, for example `MUL_MAT@token_embd.weight`.
- `MUL_MAT~<part of a weight tensor name>`, for example `MUL_MAT~ffn`.

For an iOS app, set these variables in the Xcode scheme: Product > Scheme > Edit Scheme > Run > Arguments > Environment Variables.

## Apply the patch and build

### Apply the patch

1. Run `scripts/setup.sh`.

The script gets llama.cpp at the base commit into `llama.cpp/` in this repository. Then it applies the patch. To use another directory, give it as the first argument: `scripts/setup.sh /path/to/llama.cpp`. The directory must be a clean llama.cpp checkout or must not exist. If the patch is already applied, the script does nothing. In this case, the script does not look for other local changes.

To apply the patch manually:

1. Clone llama.cpp: `git clone https://github.com/ggml-org/llama.cpp`.
2. Go into the clone: `cd llama.cpp`.
3. Check out the base commit: `git checkout 338fce6511c744893cc0d3d958b275f40552c7a4`.
4. Check the patch: `git apply --check /path/to/apple5-lfm3.patch`.
5. Apply the patch: `git apply /path/to/apple5-lfm3.patch`.

The patch applies only to this commit. On other commits, the patch can fail to apply. Then you must change the patch manually.

### Build for iOS

1. Go into the patched llama.cpp directory: `cd llama.cpp`.
2. Run `./build-xcframework.sh ios-device`.
3. Link `build-apple/llama.xcframework` into your app.

### Test on a Mac

1. Go into the patched llama.cpp directory: `cd llama.cpp`.
2. Configure a Mac build: `cmake -B build-mac -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`.
3. Build the test tool: `cmake --build build-mac --target test-backend-ops -j`.
4. Run the backend tests on the compat path: `GGML_METAL_SIMULATE_APPLE5=1 build-mac/bin/test-backend-ops test -b MTL0`.

If the configure or link step fails with `tapi error: malformed file ... unknown architecture`, the Command Line Tools SDK is newer than the Xcode linker. Run `export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)` and configure again.

## Prepare a model

The source model is `LFM3-1.7B-QAD-Q4_0.gguf` from [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF). The model has its own license. Read it on the model page.

The scripts need Python 3.10 or later and numpy. To install numpy in a virtual environment, run `python3 -m venv .venv` and `.venv/bin/pip install numpy`. Then use `.venv/bin/python` instead of `python3` in the commands below. They use the `gguf` package from the llama.cpp checkout (`gguf-py`). They look for llama.cpp in `llama.cpp/` in this repository. To use another directory, set `LLAMA_CPP_DIR`.

1. Run `scripts/setup.sh`, if you did not do it before.
2. Go to the root of this repository. Run all the commands below there.
3. Download the source model:

   ```
   curl -L -o LFM3-1.7B-QAD-Q4_0.gguf https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-QAD-Q4_0.gguf
   ```

4. Build `llama-quantize`:

   ```
   cmake -S llama.cpp -B llama.cpp/build-mac -DGGML_METAL=ON
   cmake --build llama.cpp/build-mac --target llama-quantize -j
   ```

5. Convert the token embeddings to Q8_0. The sketch script needs Q8_0 token embeddings. The other tensors stay Q4_0:

   ```
   llama.cpp/build-mac/bin/llama-quantize --allow-requantize --pure --token-embedding-type q8_0 LFM3-1.7B-QAD-Q4_0.gguf LFM3-1.7B-QAD-Q4_0-embQ8.gguf Q4_0
   ```

6. Add the output sketch:

   ```
   python3 scripts/add_output_sketch.py LFM3-1.7B-QAD-Q4_0-embQ8.gguf LFM3-1.7B-QAD-Q4_0-embQ8-sketch.gguf
   ```

7. Optional: add a draft head for speculative decoding. The last argument sets the type of the draft matrices, `q8_0` (default) or `q4_0`:

   ```
   python3 scripts/add_draft_head.py LFM3-1.7B-QAD-Q4_0-embQ8-sketch.gguf draft.safetensors LFM3-1.7B-QAD-Q4_0-embQ8-sketch-draftq4.gguf q4_0
   ```

   For `q4_0`, put `draftq4` in the output name. The chat app of lfm3-ipad selects this file first.

The safetensors file must have the F32 tensors `fc.weight` [n_embd, 2*n_embd], `blocks.0.norm.weight`, `blocks.0.gate.weight`, `blocks.0.up.weight`, `blocks.0.down.weight` and `norm.weight`. This repository does not include the training code or the training data of the draft head. Without a `draft.safetensors` file, skip step 7.

Create the llama.cpp context with flash attention disabled (`LLAMA_FLASH_ATTN_TYPE_DISABLED`). Flash attention is not available on the compat path.

## Limits

- Use Q4_0 weights. The compat Q4_0 kernels are the fastest. K-quants and iq4_xs are much slower on the compat path.
- Flash attention is not available on the compat path.
- The A12Z GPU has a working set of about 4 GB. Keep the model and the context below this size.
- The output sketch is used only for decodes with 1 to 8 output rows. Other decodes compute all logits.
- On the compat path, MUL_MAT supports only the weight types in "What the patch changes". Ops that need SIMD-group reduction and have no compat kernel stay on the CPU, for example MUL_MAT_ID, SUM_ROWS and GROUP_NORM.
- MUL_MAT on the compat path needs F32 src1, F32 output and a contiguous output. The row size must be a multiple of 32. The values in a src1 row must be contiguous.

## Measured results

The author measured these values on 2026-10-31 with the benchmark app of [lfm3-ipad](https://github.com/0xbyt4/lfm3-ipad). The lfm3-ipad README gives the commands and the prompt files to repeat the measurement.

- Device: iPad Pro 11-inch (2nd generation, A12Z), iPadOS 27.0.1.
- Model file: `LFM3-1.7B-QAD-Q4_0-embQ8-sketch-draftq4.gguf`. The file has Q4_0 weights, Q8_0 token embeddings, the output sketch and a Q4_0 draft head. With speculative decoding off, the draft head does not run.
- Decoding: greedy. In each group, the first 128 generated tokens were the same in all runs, with and without speculative decoding.
- Run length: 60 s of generation, 120 s for the mixed prompts. Before each run, the app waited for the nominal thermal state and the Mac waited 60 s.
- Run order in each group: off, on, on, off, off, on.
- Values: the mean of 3 runs. The range is in parentheses.

| Prompt | Speculative decoding off (tok/s) | Speculative decoding on (tok/s) | Change | Correct drafts |
|---|---|---|---|---|
| Essay | 38.6 (37.4-40.2) | 38.9 (37.8-40.3) | +0.8% | 33% |
| Bash script | 37.2 (36.9-37.5) | 41.3 (41.0-41.5) | +11.0% | 47% |
| Math word problem | 37.1 (37.0-37.3) | 41.1 (41.1-41.2) | +10.7% | 47% |
| 10 mixed prompts | 34.5 (33.9-34.9) | 37.0 (36.6-37.6) | +7.5% | 47% |

The prompt speed was 201 tok/s (198.5-202.9), from 72 prompts of 512 tokens in these runs.

- A verify step decodes two tokens, so it costs more than a normal decode. Speculative decoding is faster only when the draft is correct often enough. At 33% correct drafts (essay), the change is smaller than the spread of the runs.
- In the mixed prompts, 2 of the 3 runs with speculative decoding reached the "serious" thermal state. Only 1 of the 3 runs without it did. The real gain for these prompts can therefore be a little higher.
- The chat app of lfm3-ipad samples with temperature 0.1 and top-k 50. In chat, the author saw higher speeds than in the benchmark. The author did not find the cause, so this README gives only the benchmark values.
