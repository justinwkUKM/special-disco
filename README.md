# 🔒 PII Redaction Dataset Generator  
### Synthetic PII Redaction Pipeline for Fine-Tuning Small LLMs (e.g., Gemma 3 270M)

This repository provides a complete end-to-end pipeline for generating a high-quality synthetic dataset for structured PII redaction, optimized for training compact language models.

It includes:

- Clean canonical base examples  
- A hybrid Regex + Noise Mutation Engine  
- Optional teacher LLM (ChatGPT 5.1) augmentation  
- Dataset balancing tools  
- Schema validation  
- Training-ready JSONL output  

The system aligns with Distil-PII style structured output.

---

## 📁 Project Structure

```
pii_pipeline/
│
├── config.py
├── schemas.py
├── utils.py
│
├── pii_mutation_engine_v2.py
├── teacher_prompts.py
├── teacher_api.py
│
├── dataset_generator.py
├── run_pipeline.py
│
├── balance_dataset.py
├── validate_dataset.py
│
├── clean_samples/
│   └── base_clean_samples.json
│
├── outputs/
│   ├── temp/
│   └── final_dataset/
│       ├── pii_training_dataset.jsonl
│       ├── pii_training_dataset.balanced.jsonl
│       └── logs/
│
└── README.md
```

---

## 🚀 Overview

This pipeline generates a training-ready JSONL dataset for fine-tuning models to produce structured PII redaction output with mandatory fields:

- redacted_text  
- entities → value, replacement_token, reason

### Features:

- Clean base examples  
- Synthetic corruption engine  
- Teacher LLM augmentation  
- Balancing by PII type  
- Schema validation  
- High volume synthetic dataset generation  

---

## 🔧 Installation

```
git clone <repo>
cd pii_pipeline
pip install -r requirements.txt
```

Typical dependencies:

```
jsonlines
regex
python-dotenv
tqdm
openai
```

---

## 🧠 Usage

### 1️⃣ Prepare clean base samples

Edit:

```
clean_samples/base_clean_samples.json
```

Each looks like:

```json
{
  "id": "sample_001",
  "question": "Redact provided text...",
  "context": "Hi, I'm John Smith...",
  "answer": {
    "redacted_text": "Hi, I'm [PERSON]...",
    "entities": [
      { "value": "John Smith", "replacement_token": "[PERSON]", "reason": "person name" }
    ]
  }
}
```

---

### 2️⃣ Run the full pipeline

```
python run_pipeline.py
```

This generates:

```
outputs/final_dataset/pii_training_dataset.jsonl
```

---

### 3️⃣ Balance the dataset

```
python balance_dataset.py outputs/final_dataset/pii_training_dataset.jsonl                           outputs/final_dataset/pii_training_dataset.balanced.jsonl
```

---

### 4️⃣ Validate

```
python validate_dataset.py outputs/final_dataset/pii_training_dataset.balanced.jsonl
```

---

## 📦 Output Format

Each line in the JSONL dataset follows this schema:

```json
{
  "id": "uuid-or-string",
  "question": "Redact provided text...",
  "context": "Noisy user text",
  "answer": {
    "redacted_text": "Clean output with tokens",
    "entities": [
      {
        "value": "original snippet",
        "replacement_token": "[TOKEN]",
        "reason": "why it was redacted"
      }
    ]
  }
}
```

---

## 🏋️ Fine-Tuning (Gemma 270M)

Recommended hyperparameters:

| Setting | Value |
|--------|-------|
| LR | 2e-4 |
| Scheduler | cosine |
| Warmup | 3% |
| Weight decay | 0.1 |
| Batch size | 64–128 |
| Epochs | 3–5 |
| Max seq len | 1024 |
| Gradient clipping | 1.0 |

---

## 📊 Validation Metrics

- Entity-level precision & recall  
- Replacement-token correctness  
- Redacted-text equality  
- PII-class distribution  

---

## 🤝 Contributing

You can contribute by:

- Adding clean samples  
- Expanding mutation rules  
- Improving teacher prompts  
- Adding domain-specific PII types  

---

## 🧩 Roadmap

- [x] Regex mutation engine  
- [x] Teacher LLM augmentation  
- [x] Dataset balancer  
- [x] Validator  
- [ ] Multilingual PII  
- [ ] OCR noise simulation  
- [ ] Domain-specific extensions  

---

## 🛡 License

MIT License
