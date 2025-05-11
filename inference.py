from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_response(query):
    tokenizer = AutoTokenizer.from_pretrained("model")
    model = AutoModelForCausalLM.from_pretrained("model", device_map="auto")

    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
