# AI Coding Assistant Tutor

This project is an AI-powered Coding Tutor built using Hugging Face Transformers, PEFT (QLoRA), and Flask. It fine-tunes Gemma-2B on 500+ Python coding problems and serves real-time responses via a REST API.

## Features
- Fine-tuned Gemma-2B with QLoRA for low-resource training
- 4-bit quantized model for efficient inference
- Flask-based REST API for real-time code suggestions
- Handles natural queries like "Write a Python function for binary search"

## Project Structure
```
ai-coding-tutor/
├── app.py                # Flask API application
├── inference.py         # Inference logic using quantized model
├── requirements.txt     # Project dependencies
├── train.py             # Fine-tuning logic using PEFT + QLoRA
├── README.md
├── data/
│   └── sample_dataset.csv  # Sample training dataset
├── model/
│   └── load_model.py       # Model loading and quantization logic
└── utils/
    └── response_formatter.py  # Response formatting utilities
```

## Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the Flask API
```bash
python app.py
```

### Sample API Call
```
POST /predict
{
  "query": "Write a Python function for binary search"
}
```

## License
MIT
