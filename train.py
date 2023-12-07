from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
special_tokens_dict = {'additional_special_tokens': ['<|endoftext|>']}
tokenizer.add_special_tokens(special_tokens_dict)

# Load your custom dataset
dataset_file = "/content/processed_subset_dataset.txt"
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=dataset_file,
    block_size=128  # Adjust the block size as needed
)

print("Dataset loaded.")

# Prepare data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Set mlm to True if you want to train with masked language modeling
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",  # Select model path for checkpointing
    overwrite_output_dir=True,  # Overwrite existing files in the output directory
    num_train_epochs=2,  # Set number of training epochs
    per_device_train_batch_size=16,  # Set batch size
    save_steps=1000,  # Set number of updates steps before creating a checkpoint
    save_total_limit=2,  # Set total number of checkpoints to save
    prediction_loss_only=True,  # Set to True to save disk space
    logging_steps=100,  # Set number of updates steps before logging training metrics
    logging_first_step=True,  # Log the first global_step
    logging_dir="./logs",  # Directory for storing logs
    seed=42,  # Set a global random seed
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    # prediction_loss_only=True,
)

print("Training...")
# Start training
trainer.train()

# Save model
trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")