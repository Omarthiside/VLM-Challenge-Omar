# System Architecture & Technical Defense

## 1. Model Selection Defense
**Selected Model:** `Qwen2.5-VL-2B`

Given the strict hardware constraint of free-tier GPU compute (Kaggle 2x T4 32GB total, or a single 16GB T4 fallback), memory efficiency was the primary driver for model selection. 

| Model | Parameters | Est. 4-bit VRAM (Base) | Temporal Capabilities | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-VL-2B** | 2B | ~2.0 GB | Native video support, dynamic resolution | **Selected.** Fits comfortably within a single T4's 16GB limit even with 8 frames and gradient checkpointing. |
| **LLaVA-NeXT-Video-7B** | 7B | ~4.5 GB | Strong zero-shot | Rejected. Too close to OOM limits on T4 when adding LoRA adapters and activation memory for multi-frame processing. |
| **VideoLLaMA2-7B** | 7B | ~4.5 GB | SOTA spatial-temporal | Rejected. High VRAM footprint and complex training pipeline overhead for the given 36-hour timeframe. |

## 2. Frame Sampling Rationale
**Strategy:** Motion-Magnitude Adaptive Sampling

Uniform sampling (e.g., taking 1 frame every second) fails at temporal grounding because it blindly misses the micro-actions that define operation boundaries. 

Instead, this pipeline uses a **Motion-Magnitude** approach. By calculating the absolute pixel difference between consecutive frames within a $\pm0.5$ second window of the boundary, the system identifies the frames with the highest motion scores. 
* **Why it beats uniform:** The transition from "Tape" to "Put Items" involves rapid hand movement. Motion sampling guarantees the model sees the exact frames where the worker's hands change tools or items, providing the necessary visual cues for the model to learn the `anticipated_next_operation` (AA@1 metric).

## 3. Failure Mode Analysis
Based on the visual similarities in the OpenPack dataset, the model is most likely to confuse **"Tape"** with **"Pack"**.
* **Hypothesis:** Both operations involve the worker leaning over the box with their hands inside or near the flaps. If the model fails to explicitly recognize the tape dispenser in the worker's hand due to occlusion or motion blur, the spatial geometry of the two actions looks nearly identical to the vision encoder. 
* **Impact on metrics:** This visual ambiguity will artificially lower the Operation Classification Accuracy (OCA) and heavily disrupt the Anticipation Accuracy (AA@1) if the sequence grammar is broken.