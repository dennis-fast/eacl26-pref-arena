"""
Shared helpers for embedding workflows.
"""

import csv
import os
import re
import numpy as np


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).strip().lower()).strip()


def find_col(headers, aliases):
    aliases = [norm(a) for a in aliases]
    for c in headers:
        cn = norm(c)
        for a in aliases:
            if cn == a or a in cn:
                return c
    return None


def clean_text(x) -> str:
    try:
        import pandas as pd
        if pd.isna(x):
            return ""
    except Exception:
        if x is None:
            return ""
    s = str(x).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_doc(title: str, abstract: str, sep_token: str = " [SEP] ") -> str:
    t = clean_text(title)
    a = clean_text(abstract)
    if not a:
        return t
    return f"{t}{sep_token}{a}"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def load_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings not found: {path}")
    data = np.load(path, allow_pickle=True)
    ids = data["ids"].astype(str)
    emb = data["embeddings"].astype(np.float32, copy=False)
    return ids, emb


def write_ids_csv(path, ids):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "paper_id"])
        for i, pid in enumerate(ids):
            w.writerow([i, pid])


def load_titles(path):
    if not path or not os.path.exists(path):
        return {}
    titles = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("paper_id")
            title = row.get("title")
            if pid:
                titles[str(pid)] = title or ""
    return titles


def select_device(device=None):
    if device and device != "auto":
        return device
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts(texts, tokenizer, model, batch_size=16, device=None, pooling="mean"):
    import torch
    device = select_device(device)
    model.to(device)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model(**inputs)
            if pooling == "cls":
                emb = output.last_hidden_state[:, 0, :]
            else:
                emb = mean_pooling(output, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0).numpy()


def load_specter2_model(base_model, adapter_model, allow_fallback=True):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModel.from_pretrained(base_model)

    if not adapter_model:
        return tokenizer, base, False

    # Prefer adapters library if available (SPECTER2 ships as an adapter)
    try:
        from adapters import AutoAdapterModel
        adapter_model_obj = AutoAdapterModel.from_pretrained(base_model)
        adapter_name = adapter_model_obj.load_adapter(
            adapter_model,
            source="hf",
            load_as="proximity",
            set_active=True,
        )
        try:
            adapter_model_obj.set_active_adapters(adapter_name)
            if hasattr(adapter_model_obj, "enable_adapters"):
                adapter_model_obj.enable_adapters()
        except Exception:
            pass
        return tokenizer, adapter_model_obj, True
    except Exception:
        pass

    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, adapter_model)
        return tokenizer, model, True
    except Exception as err:
        if "peft_type" in str(err):
            patched = _load_adapter_with_patched_config(base, adapter_model)
            if patched is not None:
                return tokenizer, patched, True
            if allow_fallback:
                return tokenizer, base, False
        raise


def _load_adapter_with_patched_config(base_model, adapter_model):
    try:
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
        import json

        config_path = hf_hub_download(adapter_model, "adapter_config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)

        if "peft_type" not in cfg:
            if "r" in cfg and "lora_alpha" in cfg:
                cfg["peft_type"] = "LORA"
            else:
                return None

        peft_config = None
        if cfg.get("peft_type") == "LORA":
            from peft import LoraConfig
            peft_config = LoraConfig(**cfg)
        else:
            from peft import PeftConfig
            if hasattr(PeftConfig, "from_dict"):
                peft_config = PeftConfig.from_dict(cfg)

        if peft_config is None:
            return None

        try:
            return PeftModel.from_pretrained(base_model, adapter_model, config=peft_config)
        except TypeError:
            return PeftModel.from_pretrained(base_model, adapter_model, peft_config=peft_config)
    except Exception:
        return None
