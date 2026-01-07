# ComfyUI-TwinFlow

A ComfyUI custom node implementation of **TwinFlow**: Realizing One-step Generation on Large Models with Self-adversarial Flows.

This project allows you to accelerate DiT (Diffusion Transformer) models (like Qwen-Image, Z-Image, etc.) to 1-step or few-step generation using TwinFlow patch weights.

Based on the paper: [TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows](https://arxiv.org/abs/2512.05150)

Github: https://github.com/inclusionAI/TwinFlow

## Features

- **TwinFlow Model Patcher**: Loads specific TwinFlow patch weights (SafeTensors or GGUF) and injects them into the diffusion model.
- **TwinFlow Sampler/Scheduler**: Custom sampling logic (Euler/Heun) and scheduling (Kumaraswamy transform) specifically designed for TwinFlow's rectified flow.
- **TwinFlow KSampler**: An all-in-one node for ease of use.

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-username/ComfyUI-TwinFlow.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Model Patching
Place your TwinFlow patch files (e.g., `twinflow_qwen.safetensors` or `twinflow_sana.gguf`) in your `ComfyUI/models/diffusion_models/` directory.

Use the **TwinFlow Model Patcher** node:
- **model**: Connect your base DiT model (e.g., Qwen, SD3.5).
- **patch_file**: Select the corresponding TwinFlow patch file.

### 2. Sampling
You can use the **TwinFlow KSampler** for a simplified workflow, or use the individual **TwinFlow Sampler** and **TwinFlow Scheduler** nodes with the standard `SamplerCustom` node.

**Key Parameters:**
- `sampling_style`: 
  - `few`: For extremely low steps (e.g., 1-2 steps).
  - `any`: For flexible step counts.
  - `mul`: For multi-step generation.
- `sampling_method`: `euler` (1st order) or `heun` (2nd order).
- `dist_ctrl_a/b/c`: Controls the Kumaraswamy time distribution (default 1.0 is usually linear).
- `gap_start/gap_end`: Defines the time boundaries for the flow.

## Notice

ComfyUI has another Twinflow implementation as follows:

https://github.com/smthemex/ComfyUI_TwinFlow

This implementation directly integrates the Twinflow source code, but it doesn't quite conform to ComfyUI's implementation specifications. It uses a completely self-contained model loader and Ksampler internally, making it incompatible with other ComfyUI nodes. It also lacks some options found in the original Twinflow.
Our implementation is based on the Twinflow source code, but we have completely reimplemented the sampling process to fully comply with ComfyUI standards. The nodes use standard inputs and outputs, allowing you to freely combine them with other ComfyUI nodes and experiment with any LoRA models you want.

## Citations

```bibtex
@article{cheng2025twinflow,
  title={TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows},
  author={Cheng, Zhenglin and Sun, Peng and Li, Jianguo and Lin, Tao},
  journal={arXiv preprint arXiv:2512.05150},
  year={2025}
}
```

## Credits

Original implementation by [inclusionAI/TwinFlow](https://github.com/inclusionAI/TwinFlow).

## LICENSE

```text
MIT License

Copyright (c) 2025 ComfyUI-TwinFlow Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```