
import torch
import numpy as np
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# ── Load Dataset ──────────────────────────────
dataset = load_dataset("imdb")
train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
test_dataset  = dataset["test"].shuffle(seed=42).select(range(500))

# ── Tokenizer ─────────────────────────────────
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test  = test_dataset.map(tokenize_function, batched=True)

tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_test  = tokenized_test.rename_column("label", "labels")

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ── Model ─────────────────────────────────────
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)

# ── Metrics ───────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="weighted")
    return {"accuracy": round(acc, 4), "f1": round(f1, 4)}

# ── Training ──────────────────────────────────
training_args = TrainingArguments(
    output_dir="./bert-imdb-results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    report_to="none",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

trainer.train()

# ── Evaluate ──────────────────────────────────
results = trainer.evaluate()
print(results)

# ── Save ──────────────────────────────────────
model.save_pretrained("./bert-imdb-model")
tokenizer.save_pretrained("./bert-imdb-model")
print("Done!")
