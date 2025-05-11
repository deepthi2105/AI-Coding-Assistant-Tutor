from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path="model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    return tokenizer, model# Placeholder for advanced model loading with quantization
