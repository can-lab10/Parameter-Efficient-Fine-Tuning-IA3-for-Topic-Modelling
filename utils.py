import nltk
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


# Function to preprocess inputs and outputs for training
def preprocess_train(examples, tokenizer, max_input_length, max_target_length, prefix, stop_words):
    inputs = [prefix + doc for doc in examples["input"]]
    inputs = [" ".join([word for word in doc.split() if word.lower() not in stop_words]) for doc in inputs]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=examples["output"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Function to preprocess inputs for evaluation
def preprocess_eval(examples, tokenizer, max_input_length, prefix, stop_words):
    inputs = [prefix + doc for doc in examples["input"]]
    inputs = [" ".join([word for word in doc.split() if word.lower() not in stop_words]) for doc in inputs]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length", return_tensors="pt")
    return model_inputs

# Function to compute Jaccard Similarity
def compute_jaccard_similarity(predictions, references):
    mlb = MultiLabelBinarizer()
    pred_binarized = mlb.fit_transform([set(pred.split()) for pred in predictions])
    ref_binarized = mlb.transform([set(ref.split()) for ref in references])
    jaccard_scores = [jaccard_score(pred, ref, average='binary') for pred, ref in zip(pred_binarized, ref_binarized)]
    return jaccard_scores
