import json
from pathlib import Path


def test_notebook_is_valid_and_qwen_only():
    path = Path("qlora_finetune.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for required in [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "BitsAndBytesConfig",
        "LoraConfig",
        "prepare_model_for_kbit_training",
        "Trainer",
        "matplotlib",
        "push_to_hub",
        "HF_TOKEN",
    ]:
        assert required in source
    assert "llama" not in source.lower()
    assert "meta-llama" not in source.lower()
