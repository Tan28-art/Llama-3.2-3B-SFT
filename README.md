# LLMA-O: AI Assistant with Llama 3.2-3B

## Table of Contents
1.  [Introduction](#introduction)
2.  [Key Features](#key-features)
3.  [Architecture](#architecture)
    * [Model](#model)
    * [Fine-Tuning](#fine-tuning)
    * [Inference](#inference)
    * [Backend & Client](#backend--client)
4.  [Project Structure](#project-structure)
5.  [Setup and Usage](#setup-and-usage)
    * [Prerequisites](#prerequisites)
    * [Model Setup](#model-setup)
    * [Server Setup](#server-setup)
    * [Client Setup](#client-setup)
6.  [Training Details](#training-details)
7.  [Performance](#performance)
8.  [Explored Techniques](#explored-techniques)
9.  [License](#license)

## Introduction
This project focuses on utilizing the Llama 3.2-3B language model to create a capable AI assistant. The assistant is designed to understand and respond to user queries effectively by leveraging various performance improvement techniques, including Supervised Fine-Tuning (SFT), Self-Consistency for inference, and a robust backend system for deployment. The core of the project involves fine-tuning the Llama model on a diverse set of datasets to enhance its instruction following, safety alignment, mathematical reasoning, and general question-answering capabilities.

## Key Features
* **Fine-tuned Llama 3.2-3B Model:** Leverages a powerful base model fine-tuned for enhanced assistant capabilities.
* **Instruction Following:** Trained to accurately follow user instructions and prompts.
* **Safety Alignment:** Incorporates safety measures through dataset selection and fine-tuning.
* **Mathematical Reasoning:** Enhanced capabilities for solving math-related problems.
* **Versatile Question-Answering:** Capable of answering questions across various domains.
* **FastAPI Backend:** A simple and efficient backend server for model deployment.
* **vLLM for Accelerated Inference:** Significantly speeds up model response times.
* **Self-Consistency:** Employs ROUGE-L similarity-based majority voting (from 5 samples) to improve response quality and reliability.
* **Contextual Conversations:** The backend is designed to store conversation context for more coherent interactions.

## Architecture

### Model
* **Base Model:** `meta-llama/Llama-3.2-3B`

### Fine-Tuning
* **Method:** Supervised Fine-Tuning (SFT) using QLoRA to manage computational resources.
* **Datasets:** A combination of publicly available datasets were used for fine-tuning, subsetted to a maximum of 10,000 samples each, resulting in a final sample size of 88,031. Key datasets include:
    * `tatsu-lab/alpaca` (General instruction following)
    * `databricks/databricks-dolly-15k` (General instruction following)
    * `Anthropic/hh-rlhf` (Safety alignment)
    * `gsm8k` (Maths and reasoning)
    * `meta-math/MetaMathQA` (Maths and reasoning)
    * `hotpot_qa` (Various question-answering formats)
    * `cais/mmlu` (Various question-answering formats)
    * `commonsense_qa` (Various question-answering formats)
    * `trivia_qa` (Various question-answering formats)
    * `domenicrosati/TruthfulQA` (Truthfulness and safety)
* **Prompt Format:** A standardized prompt format was crucial for consistent model learning:
    ```
    Instruction:
    [Instruction/Question Text]

    ### Input:
    [Optional Context/Input Data]

    ### Response:
    [Target Answer/Completion]<|end_of_text|>
    ```
    Separate formatting functions were created for each dataset to ensure data consistency.

### Inference
* **Engine:** `vLLM` is used for optimized and accelerated inference, significantly reducing processing time.
* **Self-Consistency:** The model generates 5 candidate responses using temperature sampling. The final response is selected based on ROUGE-L similarity (majority voting), enhancing reliability.
    * Sampling Parameters (default): `n=5`, `temperature=0.7`, `top_p=0.9`, `max_new_tokens=512`.

### Backend & Client
* **Backend:**
    * Built with `FastAPI` for simplicity and development convenience.
    * Deployed in a chat mode, storing conversation context.
    * The server script is `LLM_Server_Final.py`.
* **Client:**
    * A Python script (`LLM_Client_Final.py`) that allows users to interact with the FastAPI server.

## Project Structure

.
├── LLM_Client_Final.py         # Client script to interact with the server
├── LLM_Server_Final.py         # FastAPI server script for model deployment
├── Llama3.2-3B-final-fine-tune.ipynb # Jupyter notebook for fine-tuning
├── README.md                   # This file
└── models/
└── llama3.2-3b-sft-merged-2048sl/ # Directory for the fine-tuned model (see Model Setup)

## Setup and Usage

### Prerequisites
* Python 3.8+
* Access to a machine with GPU(s) compatible with vLLM and PyTorch (e.g., NVIDIA A100 for training/vLLM).
* Familiarity with setting up Python environments.

### Model Setup
1.  **Base Model:** You need access to the `meta-llama/Llama-3.2-3B` model.
2.  **Fine-tuned Model:** The fine-tuned model used in this project is `llama3.2-3b-sft-merged-2048sl`.
    * **Option A (Download):** If provided, download the fine-tuned model files and place them in a directory (e.g., `./models/llama3.2-3b-sft-merged-2048sl/`).
    * **Option B (Reproduce):** Run the `Llama3.2-3B-final-fine-tune.ipynb` notebook. This will generate the adapters. You will then need to merge these adapters with the base model to get the `llama3.2-3b-sft-merged-2048sl` model. Ensure the output path in the notebook corresponds to where the server expects the model.
3.  **Configure Model Path:** In `LLM_Server_Final.py`, update the `MERGED_MODEL_PATH` variable to point to the location of your fine-tuned model:
    ```python
    MERGED_MODEL_PATH = "./models/llama3.2-3b-sft-merged-2048sl" # Or your actual path
    ```

### Server Setup
1.  **Create a Python environment and install dependencies:**
    ```bash
    pip install -r requirements_server.txt
    ```
    Key dependencies include `fastapi`, `uvicorn`, `vllm`, `transformers`, `torch`, `evaluate` (for ROUGE).
2.  **Run the FastAPI server:**
    ```bash
    python LLM_Server_Final.py
    ```
    By default, the server starts on `0.0.0.0:8004`. You can modify the host and port in the `uvicorn.run` call at the end of the script if needed.

### Client Setup
1.  **Create a Python environment and install dependencies:**
    ```bash
    pip install -r requirements_client.txt
    ```
    Key dependency is `requests`.
2.  **Run the client:**
    ```bash
    python LLM_Client_Final.py [--host <server_host>] [--port <server_port>]
    ```
    Example:
    ```bash
    python LLM_Client_Final.py --host 127.0.0.1 --port 8004
    ```
    If the server is running on the default host and port, you can simply run `python LLM_Client_Final.py`.
    Type `exit` or `quit` to end the chat.

## Training Details
The model was fine-tuned using the `Llama3.2-3B-final-fine-tune.ipynb` notebook.
* **Environment:** Utilized ASU's university cluster (Sol) with NVIDIA A100 GPUs (80GB VRAM).
* **Technique:** QLoRA for efficient fine-tuning.
* **Libraries:** `trl`, `peft`, `bitsandbytes`, `transformers`, `datasets`, `accelerate`.
* **Training Parameters:**
    * `max_seq_length = 512`
    * `train_batch_size = 32` (effective, via gradient accumulation)
    * `num_train_epochs = 1`
    * Optimizer: `paged_adamw_32bit`
    * Learning Rate: `2e-4` with a cosine scheduler.
* **Training Time:** Approximately 2 hours and 20 minutes with an effective batch size of 32.

(For detailed steps, refer to the `Llama3.2-3B-final-fine-tune.ipynb` notebook.)

## Performance
* **Inference Speed:** Using vLLM, inference for 27,500 prompts (with self-consistency, 5 samples each) took approximately 17 minutes, a significant improvement over the estimated 3 hours with a traditional Hugging Face pipeline.
* **Training Metrics:**
    * `eval_samples`: 800
    * `total_flos`: 456366752067889150
    * `train_loss`: 1.537569...
    * `train_runtime`: 8087.8856 seconds
    * `train_samples_per_second`: 10.884
    * `train_steps_per_second`: 0.34
    (Full results can be found in the project's shared Google Drive, as mentioned in the report.)

## Explored Techniques
The following techniques were explored during the project:
* **Direct Preference Optimization (DPO):** Attempted with the `argilla/ultrafeedback-binarized-preferences` dataset. However, it did not yield functional results post-training, with the model failing basic interactions.
* **Few-Shot Prompting (Static and Dynamic):** Found to be ineffective or detrimental. The Llama SFT model used as a dynamic classifier performed poorly. We were limited to only using one model so our results are probably not the norm and few shot prompting in general should improve model performance.
* **Chain-of-Thought Prompting:** Made answers unnecessarily verbose and created more problems than it solved, as the chosen datasets for math/reasoning (gsm8k, metamathqa) already enabled good generalization for those tasks.
* **Self-Consistency Methods:**
    * **RogueL + Math based majority voting:** Led to incorrect identification of math questions.
    * **Cosine similarity based majority voting:** Faced issues similar to dynamic few-shot prompting, often choosing hallucinated answers.
    * **RogueL similarity based majority voting (Chosen Method):** Performed the best with not as many unintended behaviors.
