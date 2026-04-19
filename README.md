# BERT Fine-tuned — IMDB Sentiment Analysis

## About
Fine-tuned `bert-base-uncased` on IMDB dataset for sentiment analysis.

## Model
Model weights available on HuggingFace:
👉 https://huggingface.co/your-username/bert-imdb-sentiment

## Results
| Metric   | Score |
|----------|-------|
| Accuracy | ~88%  |
| F1 Score | ~0.88 |

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py
```

## Inference
```python
from transformers import pipeline

classifier = pipeline(
    'sentiment-analysis',
    model='your-username/bert-imdb-sentiment'
)
result = classifier("This movie was amazing!")
print(result)
```

## Labels
- LABEL_0 → Negative
- LABEL_1 → Positive
