import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import torch
from peft import PeftModel
from nltk.corpus import stopwords
import nltk
from utils import preprocess_eval, compute_jaccard_similarity

# Define the main evaluation function
def evaluate(model_name, adapter_dir, data_path, max_input_length=1024, max_target_length=128, output_path="eval_results/evaluation_results.xlsx"):
    stop_words = set(stopwords.words('english'))

    # Load evaluation data
    data = pd.read_csv(data_path)

    # Import the model and attach the adapter
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    prefix = "topics"
    
    predictions = []
    references = list(data['output'])
    
    # Loop throuh the data and generate output for each input
    for i in range(len(data)):
        inputs = preprocess_eval({'input': [data.loc[i, 'input']]}, tokenizer, max_input_length, prefix, stop_words)
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        
        with torch.no_grad():
            generated_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_target_length, num_beams=5)
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        predictions.append(decoded_preds[0])
    
    # Add the predictions to the data frame
    data['Generated Output'] = predictions

    # Compute the Jaccard scores of the prediction by comparing them to the ground truth
    jaccard_scores = compute_jaccard_similarity(predictions, references)
    data['Jaccard Similarity'] = jaccard_scores

    # Get the average Jaccard score of the evaluation data and print it
    average_jaccard_score = np.mean(jaccard_scores)
    print(f"Average Jaccard Similarity: {average_jaccard_score:.4f}")

    # Convert the data frame to excel and save it to the folder eval_results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_excel(output_path, index=False)

    print("Evaluation completed. Results saved to:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model for topic generation.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for evaluation.")
    parser.add_argument("--adapter_dir", type=str, required=True, help="Directory of the adapter checkpoints.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the evaluation data CSV file.")
    parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum input length for tokenization.")
    parser.add_argument("--max_target_length", type=int, default=128, help="Maximum target length for tokenization.")
    parser.add_argument("--output_path", type=str, default="eval_results/evaluation_results.xlsx", help="Path to save the evaluation results.")

    args = parser.parse_args()
    
    # Run the evaluation funciton
    evaluate(
        model_name=args.model_name,
        adapter_dir=args.adapter_dir,
        data_path=args.data_path,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        output_path=args.output_path
    )
