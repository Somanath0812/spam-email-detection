import pandas as pd
import torch
import mlflow
import mlflow.pytorch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==============================
# Configuration
# ==============================

model_path = "backend/ml/models/bert_spam_model"

batch_size = 8
learning_rate = 2e-5
epochs = 1


# ==============================
# Load test dataset
# ==============================

test_df = pd.read_csv("data/test.csv")

print("Testing data:", test_df.shape)


# ==============================
# Load tokenizer
# ==============================

tokenizer = AutoTokenizer.from_pretrained(
    model_path
)

print("Tokenizer loaded successfully")


# ==============================
# Dataset class
# ==============================

class EmailDataset(Dataset):

    def __init__(self, dataframe, tokenizer):

        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label_num"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, index):

        encoding = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        }


# ==============================
# Create test dataset
# ==============================

test_dataset = EmailDataset(
    test_df,
    tokenizer
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("Test samples:", len(test_dataset))


# ==============================
# Load trained BERT model
# ==============================

model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)

print("Trained BERT model loaded successfully")


# ==============================
# Device
# ==============================

device = torch.device("cpu")

model.to(device)
model.eval()

print("Using device:", device)


# ==============================
# MLflow configuration
# ==============================

mlflow.set_experiment("Spam_Email_BERT")


# ==============================
# Start MLflow run
# ==============================

with mlflow.start_run():

    # Log parameters

    mlflow.log_param(
        "model",
        "BERT"
    )

    mlflow.log_param(
        "dataset",
        "spam_mails_dataset"
    )

    mlflow.log_param(
        "batch_size",
        batch_size
    )

    mlflow.log_param(
        "learning_rate",
        learning_rate
    )

    mlflow.log_param(
        "epochs",
        epochs
    )

    mlflow.log_param(
        "max_length",
        128
    )


    # ==============================
    # Make predictions
    # ==============================

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch_number, batch in enumerate(test_loader):

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            if (batch_number + 1) % 25 == 0:

                print(
                    f"Processed "
                    f"{batch_number + 1}/"
                    f"{len(test_loader)} "
                    f"test batches"
                )


    # ==============================
    # Calculate metrics
    # ==============================

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions
    )

    recall = recall_score(
        all_labels,
        all_predictions
    )

    f1 = f1_score(
        all_labels,
        all_predictions
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )


    # ==============================
    # Log metrics to MLflow
    # ==============================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )


    # ==============================
    # Display results
    # ==============================

    print()

    print(
        "========== BERT EVALUATION RESULTS =========="
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print()

    print("Confusion Matrix:")

    print(cm)

    print()

    print(
        "MLflow tracking completed successfully"
    )

    print(
        "Run ID:",
        mlflow.active_run().info.run_id
    )