# 🚀 Custom SME AI Agents (Mistral & Llama 3)

Engineering 3 differents custom AI Agents for SME use cases (proofreading, domain-specific translation)

Deploying memory-efficient, fine-tuned LLMs for Small and Medium Enterprises (SMEs) to automate proofreading and domain-specific translation.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

## 📖 Overview
This project demonstrates a complete end-to-end pipeline for fine-tuning open-weights models (**Mistral 7B / Llama 3**) and deploying them as a scalable API. 

**Key Achievements:**
* **Memory Efficiency:** Utilized **LoRA (Low-Rank Adaptation)** and 4-bit quantization to fine-tune models on consumer-grade hardware.
* **Accuracy:** Achieved **97% syntactic accuracy** for domain-specific tasks.
* **Production-Ready:** Packaged the inference engine into a Dockerized **FastAPI** microservice.

## 🏗️ System Architecture
![System Architecture](assets/AI-Agents-SME.drawio.png)
   

## 📂 Project Structure

```text
ai-agents-sme/
├── app/
│   ├── main.py             # FastAPI application entry point
│   ├── api/
│   │   └── routes.py       # API endpoints definition
│   ├── core/
│   │   ├── config.py       # Environment variables & settings
│   │   └── inference.py    # LLM loading and generation logic (LoRA + Base Model)
│   └── prompts/
│       └── templates.py    # Domain-specific prompt engineering
├── scripts/
│   └── train_lora.py       # PEFT/LoRA fine-tuning script
├── assets/                 # Architecture diagrams
├── .gitignore              # Ignores large model weights and .env
├── Dockerfile              # Containerization instructions
├── requirements.txt        # Python dependencies
└── README.md
```
## ⚙️ How to Run (Quickstart)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Leandredt/ai-agents-sme.git](https://github.com/Leandredt/ai-agents-sme.git)
   cd ai-agents-sme