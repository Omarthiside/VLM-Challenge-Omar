# AI Agent Development Log

| Phase / Task | Tool Used | Prompt / Request | Outcome & Time Saved | Commit Hash |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Deployment** | Gemini 3.1 Pro | "Generate FastAPI & Docker boilerplate for Qwen2.5-VL deployment" | Accepted full boilerplate. Skipped manual Docker CUDA configuration. Saved: 45 mins. | `f372368` |
| **Phase 2: Data Pipeline** | Gemini 3.1 Pro | "Write data pipeline using motion-magnitude sampling to extract boundary clips." | Code accepted. Used dummy video generation to bypass 50GB local download requirement. Saved: 2 hours. | `b40beb4` |
| **Phase 3: Fine-Tuning** | Gemini 3.1 Pro | "Calculate VRAM math for Kaggle T4 notebook using Qwen2.5-VL with 8 frames." | Output matched Kaggle execution (2.31 GB). Configured training args with gradient checkpointing. Saved: 1 hour. | `9a2aa56` |
| **Phase 4: Evaluation** | Gemini 3.1 Pro | "Write evaluation script to calculate OCA, tIoU@0.5, and AA@1 metrics." | Script accepted and run locally to generate required results.json schema. Saved: 45 mins. | `46e501f` |
| **Phase 5: Architecture** | Gemini 3.1 Pro | "Draft ARCHITECTURE.md defense and format AI usage logs." | Accepted Markdown formatting and technical defense for model selection. Saved: 30 mins. | `03daa55` |
| **Code Cleanup** | Manual/AI | "Review file structure for final submission." | Manually removed auto-generated code comments to improve readability and code style. | `a62be5a` |
| **Log Sync 1** | Gemini 3.1 Pro | "Update agent log to reflect actual repository commits." | Updated log to include missing commits. | `9981749` |
| **Log Sync 2 & Diagram** | Gemini 3.1 Pro | "Add ASCII diagram to architecture and finalize exact commit hashes." | Successfully injected temporal sampling diagram into defense file. | `905b8f4` |