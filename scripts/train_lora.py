cat > scripts/train_lora.py << 'EOL'
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import yaml
import os

# Charger la configuration
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Charger le modèle de base
model_name = config["mistral"]["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

# Configurer LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Appliquer LoRA au modèle
model = get_peft_model(model, lora_config)

# Charger le dataset
dataset = load_dataset("json", data_files=os.path.join(config["paths"]["datasets"], "dataset.jsonl"), split="train")

# Tokenizer
def tokenize_function(examples):
    return tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir=config["mistral"]["lora_dir"],
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
)

# Entraînement
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    dataset_text_field="input",
    max_seq_length=512,
    tokenizer=tokenizer,
)
trainer.train()

# Sauvegarder le modèle LoRA
model.save_pretrained(config["mistral"]["lora_dir"])
print(f"Modèle LoRA sauvegardé dans {config['mistral']['lora_dir']}")

EOL
