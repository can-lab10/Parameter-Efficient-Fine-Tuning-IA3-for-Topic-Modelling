# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords

# Command to run the training script (can be overridden)
CMD ["python", "train.py", "--model_name", "google-t5/t5-small", "--data_path", "data/training_data.csv", "--output_dir", "training_results", "--epochs", "3", "--batch_size", "16", "--learning_rate", "3e-3", "--max_input_length", "1024", "--max_target_length", "128"]

