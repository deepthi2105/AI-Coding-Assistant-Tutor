# AI Coding Assistant Tutor

This project is an AI-powered Coding Tutor built using Hugging Face Transformers, PEFT (QLoRA), and Flask. It fine-tunes Gemma-2B on 500+ Python coding problems and serves real-time responses via a REST API.

## Features
- Fine-tuned Gemma-2B with QLoRA for low-resource training
- 4-bit quantized model for efficient inference
- Flask-based REST API for real-time code suggestions
- Handles natural queries like "Write a Python function for binary search"
