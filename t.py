from transformers import AutoModelForMaskedLM
AutoModelForMaskedLM.from_pretrained("roberta-large", cache_dir="hf_cache/huggingface/hub")
