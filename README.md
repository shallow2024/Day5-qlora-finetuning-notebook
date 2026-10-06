# Day 5 — Parameter-Efficient Fine-Tuning (QLoRA)

A self-contained Google Colab notebook that fine-tunes `meta-llama/Llama-3.2-1B` on a tiny technical Q&A dataset using 4-bit quantization and LoRA adapters, plots training/evaluation loss, runs a post-training inference check, and uploads adapter weights to the Hugging Face Hub.

## Open in Google Colab

After cloning or downloading this repository, open [`qlora_finetune.ipynb`](qlora_finetune.ipynb) in Google Colab.

## Requirements

- Google Colab GPU runtime, preferably T4 or better.
- Hugging Face account.
- Access approval for the gated `meta-llama/Llama-3.2-1B` model and acceptance of Meta's license.
- A Hugging Face token with write permission if uploading adapters.

The notebook uses free cloud compute when available, but Colab GPU availability, session limits, and Hugging Face provider/model access can change. The notebook does not require Groq.

## Notebook workflow

```text
GPU check
→ install Transformers / PEFT / bitsandbytes / datasets
→ authenticate to Hugging Face
→ build mini Q&A Dataset
→ tokenize prompts
→ load Llama 3.2 1B in 4-bit NF4
→ prepare model for k-bit training
→ attach LoRA adapters
→ train with Trainer
→ plot loss
→ run inference
→ upload adapter weights to Hugging Face Hub
```

## Important model-access note

`meta-llama/Llama-3.2-1B` is gated. Before running the model-loading cell:

1. Open the model page: <https://huggingface.co/meta-llama/Llama-3.2-1B>
2. Request or accept access as required.
3. Accept the model license.
4. Add `HF_TOKEN` to Colab Secrets, or use the `login()` widget in the notebook.

The token must not be committed to GitHub. The notebook never prints it.

## What QLoRA means

- **Quantization:** loads the frozen base model using 4-bit NF4 weights to reduce GPU memory use.
- **LoRA:** trains small low-rank adapter matrices instead of updating every base-model parameter.
- **QLoRA:** combines 4-bit loading with LoRA training.

The Hub upload contains adapter weights and tokenizer files. It does not upload a full copy of the base model.

## Dataset

The notebook contains eight small technical Q&A examples covering API gateways, secrets, indexes, RAG, health checks, retries, least privilege, and unit tests. This is intentionally a learning dataset, not a production training corpus. Replace it with data you are authorized to use.

## Expected result

The notebook should display:

- CUDA GPU information.
- Trainable-parameter counts showing that only a small fraction is trainable.
- Training and evaluation logs.
- A matplotlib loss plot.
- A generated answer to a test question.
- A Hugging Face Hub URL after upload.

## Limitations

- A tiny dataset can cause memorization and does not prove general quality.
- The base model remains gated for downstream users.
- Training and inference quality depend on GPU memory, package versions, prompt format, and dataset quality.
- Always evaluate the fine-tuned model against a held-out set and the unfine-tuned base model before making claims about improvement.

## License

The notebook code is MIT-licensed. The Llama model remains subject to Meta's Llama Community License and Hugging Face access terms.
