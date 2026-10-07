# Day 5 — Qwen Parameter-Efficient Fine-Tuning (QLoRA)

A self-contained Google Colab notebook that fine-tunes **`Qwen/Qwen2.5-1.5B-Instruct`** on a small technical Q&A dataset using 4-bit quantization and LoRA adapters. It records training logs, plots training/evaluation loss, runs a post-training inference test, and optionally uploads adapter weights to the Hugging Face Hub.

This version uses **Qwen** and is designed for direct access through Hugging Face.

## Open in Google Colab

Open [`qlora_finetune.ipynb`](qlora_finetune.ipynb) in Google Colab. You can upload the notebook through **File → Upload notebook**, or use the GitHub-to-Colab workflow.

## Requirements

- Google Colab GPU runtime, preferably T4 or better.
- Hugging Face account only if you want to upload adapters.
- A Hugging Face token with write permission for Hub upload.
- No Groq API key or gated-model approval is required.

The Qwen base model is publicly available on Hugging Face. Colab GPU availability, session limits, and package versions can change.

## Model

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Qwen2.5-1.5B-Instruct is an instruction-tuned causal language model suitable for a compact learning experiment. The notebook uses it as a base model, loads it in 4-bit NF4, freezes it, and trains only LoRA adapter parameters.

Model page: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>

## Notebook workflow

```text
GPU check
→ install Transformers / PEFT / bitsandbytes / datasets
→ optional Hugging Face authentication
→ build mini technical Q&A dataset
→ format Qwen instruction prompts
→ tokenize prompts
→ load Qwen2.5 1.5B in 4-bit NF4
→ prepare model for k-bit training
→ attach LoRA adapters
→ train with Trainer
→ plot training and evaluation loss
→ run Qwen inference test
→ optionally upload adapter weights to Hugging Face Hub
```

## Authentication

Downloading the public Qwen model normally works without a token. To upload your adapter:

1. Create a Hugging Face token with write permission: <https://huggingface.co/settings/tokens>
2. Add it to Google Colab Secrets with the exact name `HF_TOKEN`.
3. Run the optional authentication cell.
4. Change `HUB_REPO_ID` to your own Hub username and repository name.

Never put a real token in Python code, notebook output, or GitHub.

## What QLoRA means

- **Quantization:** loads the frozen Qwen base model using 4-bit NF4 weights to reduce GPU memory use.
- **LoRA:** trains small low-rank adapter matrices instead of updating every Qwen parameter.
- **QLoRA:** combines 4-bit loading with LoRA training.

The Hub upload contains adapter weights and tokenizer files. It does not upload a full copy of Qwen.

## Dataset

The notebook contains eight transparent technical Q&A examples covering API gateways, secrets, indexes, RAG, health checks, retries, least privilege, and unit tests. This is intentionally a learning dataset, not a production training corpus. Replace it with data you are authorized to use.

## How to run

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the cells from top to bottom.
4. Confirm that `CUDA available: True` appears.
5. Keep `MODEL_ID` as `Qwen/Qwen2.5-1.5B-Instruct`.
6. Run training and inspect the loss plot.
7. Change `test_question` to test another technical question.
8. Only after the inference test succeeds, configure `HUB_REPO_ID` and upload the adapter.

## Expected result

The notebook should display:

- CUDA GPU information.
- Trainable-parameter counts showing that only a small fraction is trainable.
- Training and evaluation logs.
- A matplotlib loss plot.
- A generated Qwen answer to a test question.
- A Hugging Face Hub URL after an optional upload.

## Limitations

- A tiny dataset can cause memorization and does not prove general quality.
- The fine-tuned Qwen model may repeat training examples instead of generalizing.
- Training and inference quality depend on GPU memory, package versions, prompt format, and dataset quality.
- Always evaluate the fine-tuned model against a held-out set and the unfine-tuned Qwen base model before making claims about improvement.
- Qwen's model and tokenizer license terms apply to use and redistribution.

## Portfolio value

This project demonstrates practical understanding of:

- Hugging Face Transformers.
- 4-bit NF4 quantization.
- PEFT and LoRA adapters.
- QLoRA training on a cloud GPU.
- Training/evaluation loss analysis.
- Post-fine-tuning inference.
- Adapter-only model publishing.

## License

The notebook code is MIT-licensed. The Qwen model and tokenizer remain subject to their own Hugging Face repository license and terms.
