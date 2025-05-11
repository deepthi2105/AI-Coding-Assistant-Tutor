from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TextDataset, DataCollatorForLanguageModeling
import torch

model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, peft_config)

# Dummy dataset load
train_dataset = TextDataset(tokenizer=tokenizer, file_path="data/sample_dataset.csv", block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    evaluation_strategy="no",
    num_train_epochs=3,
    save_steps=500,
    logging_dir="./logs",
    learning_rate=2e-4,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
