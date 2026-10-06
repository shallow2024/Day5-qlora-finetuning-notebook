import json
from pathlib import Path


def test_notebook_is_valid_and_contains_required_sections():
    path = Path("qlora_finetune.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for required in [
        "meta-llama/Llama-3.2-1B",
        "BitsAndBytesConfig",
        "LoraConfig",
        "prepare_model_for_kbit_training",
        "Trainer",
        "matplotlib",
        "push_to_hub",
        "HF_TOKEN",
    ]:
        assert required in source
