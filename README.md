# comfyui-FVMtools

Unified ComfyUI custom node pack for face detection, matching, and LoRA-based detailing.

## Nodes

All nodes appear under **FVM Tools/Face** in the ComfyUI menu.

| Node | Description |
|------|-------------|
| **Person Selector (Match)** | Single-reference face matching using InsightFace embeddings |
| **Person Selector Multi** | Multi-reference batch matching with PERSON_DATA output |
| **Person Detailer** | Per-slot LoRA + inpaint detailing pipeline |
| **Detail Daemon Options** | Optional fine-tuning for Detail Daemon parameters |
| **Inpaint Options** | Optional advanced inpaint and per-slot settings |

## Required Models

### InsightFace (Face Detection & Embedding)

Downloaded automatically on first use. Stored in:
```
ComfyUI/models/insightface/models/buffalo_l/
```

### BiSeNet (Face/Head Segmentation)

**File:** `parsing_bisenet.pth`

Place in one of these directories:
```
ComfyUI/models/gfpgan/parsing_bisenet.pth
ComfyUI/models/facedetection/parsing_bisenet.pth
```

Download: [parsing_bisenet.pth](https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth)

This is a BiSeNet face parsing model (19 semantic classes) used for generating face and head masks.

### SAM (Body Segmentation) — Optional

Required only for **body** mask mode. Uses the SAM model loaded via [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)'s **SAMLoader** node.

**Recommended model:** `sam2.1_hiera_large.pt`

Place in:
```
ComfyUI/models/sams/sam2.1_hiera_large.pt
```

Download: [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

Connect the SAMLoader output to Person Selector Multi's `sam_model` input.

## Basic Workflow

```
Image Batch → Person Selector Multi → Person Detailer → Output Images
                    ↑                        ↑
              Reference Images         Model / CLIP / VAE
              + SAM Model             + LoRAs & Prompts
```

1. **Person Selector Multi** detects faces, matches them to references, generates masks, and outputs `PERSON_DATA`
2. **Person Detailer** iterates over each enabled reference slot, applies the slot's LoRA and prompt, inpaints the masked face region, and stitches it back

## Z-Image Turbo Compatibility

Person Detailer automatically detects Z-Image Turbo (Lumina2) models and converts LoRAs trained with diffusers-style trainers on the fly. Z-Image Turbo uses fused QKV attention layers, while standard LoRAs have separate `to_q`/`to_k`/`to_v` projections — these are automatically concatenated into the fused format when needed.

No user action required. The conversion is logged in the console:
```
[FVMTools] Auto-converting LoRA 'my_lora.safetensors' for Z-Image Turbo QKV format
```

## Dependencies

```
insightface>=0.7.3
onnxruntime-gpu>=1.17.0
opencv-python>=4.8.0
numpy>=1.24.0
```

## Credits & References

This project builds on the work of several open-source projects:

| Component | Author / Project | License |
|-----------|-----------------|---------|
| Face detection & embeddings | [InsightFace](https://github.com/deepinsight/insightface) (Jia Guo et al.) | MIT |
| BiSeNet face parsing | [facexlib](https://github.com/xinntao/facexlib) (Xintao Wang) | BSD |
| Detail Daemon concept | [ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon) (Jonseed) | MIT |
| Z-Image Turbo QKV conversion | [Comfyui-ZiT-Lora-loader](https://github.com/capitan01R/Comfyui-ZiT-Lora-loader) (capitan01R) | MIT |
| SAM2 segmentation | [Segment Anything 2](https://github.com/facebookresearch/sam2) (Meta AI) | Apache 2.0 |
| SAM integration | [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) (ltdrdata) | GPL-3.0 |

## License

MIT
