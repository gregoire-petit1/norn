"""Simplified training script for a sentiment classifier."""

# This is a mock training script — the model isn't actually trained.
# The agent should use this + metrics.json to write the model card.

MODEL_NAME = "sentiment-bert-small"
TASK = "binary sentiment classification"
DATASET = "IMDB Movie Reviews (50k samples, 80/20 split)"
ARCHITECTURE = "DistilBERT fine-tuned with a classification head"
TRAINING_EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 32

if __name__ == "__main__":
    print(f"Training {MODEL_NAME} on {DATASET}")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Epochs: {TRAINING_EPOCHS}, LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}")
    print("Training complete (mock).")
