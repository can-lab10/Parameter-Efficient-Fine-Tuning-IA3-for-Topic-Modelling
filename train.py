import argparse
import nltk
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import ( AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer)
from peft import get_peft_model, IA3Config
from evaluate import load
from nltk.corpus import stopwords
from datetime import datetime
import os
from utils import preprocess_train

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model for topic modeling")
    parser.add_argument("--model_name", type=str, default="google-t5/t5-small", help="Model name to use for fine-tuning")
    parser.add_argument("--data_path", type=str, default="data/training_data.csv", help="Path to the training data CSV file")
    parser.add_argument("--output_dir", type=str, default="models/saved_model", help="Output directory for the trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-3, help="Learning rate for training")
    parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum input length for tokenization")
    parser.add_argument("--max_target_length", type=int, default=128, help="Maximum target length for tokenization")
    return parser.parse_args()

def main():
    args = parse_args()

    # Import the stop words
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Import the training data
    data = pd.read_csv(args.data_path)
    data = Dataset.from_pandas(data)

    # Import the model and its tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

    # Create a peft config of IA3 and merge it with the model
    peft_config = IA3Config(task_type="SEQ_2_SEQ_LM")
    model = get_peft_model(model, peft_config)

    prefix = "topics:"

    # Map the training data with the preprocess function in order to prepare the data for the training
    tokenized_datasets = data.map(
        lambda x: preprocess_train(x, tokenizer, args.max_input_length, args.max_target_length, prefix, stop_words), batched=True)
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )
    # Create a trainer for sequence to sequence fine-tuning
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    # Determine the training timestamp to include it in the folder name of the adapter checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    trainer.train()

    # Save the adapter and tokenizer of the fine-tuned model
    adapter_dir = os.path.join(args.output_dir, f"adapter_checkpoints_{timestamp}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

if __name__ == "__main__":
    main()
