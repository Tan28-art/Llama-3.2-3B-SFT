# llm_server_vllm.py
import os
import gc
import time
import re
import json
import numpy as np
from collections import Counter

# Server Imports
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
import uvicorn

# vLLM for Inference
from vllm import LLM, SamplingParams
import torch

# Tokenizer & ROUGE
from transformers import AutoTokenizer
import evaluate

print("Initializing LLM Server (vLLM Backend + ROUGE SC)")

# Configuration
MERGED_MODEL_PATH = "./llama3.2-3b-sft-merged-2048sl" # Path to merged SFT model
DEFAULT_N_SAMPLES = 5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_NEW_TOKENS = 512
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.95

# Global Variables
llm_engine = None
tokenizer_for_prompting = None
rouge_metric = None
create_prompt_func = None

# Pydantic Model for Request Body
class QueryRequest(BaseModel):
    user_query: str

# Helper Functions
def create_prompt_base(instruction, input_text=None, output="", include_eos=True):
    """Creates the basic prompt structure."""
    instruction_str = str(instruction).strip() if instruction is not None else "[No Instruction]"
    input_text_str = str(input_text).strip() if input_text is not None else ""
    output_str = str(output) if output is not None else ""
    global tokenizer_for_prompting
    if input_text_str:
        prompt = f"""Instruction:\n{instruction_str}\n\nInput:\n{input_text_str}\n\nResponse:\n{output_str}"""
    else:
        prompt = f"""Instruction:\n{instruction_str}\n\nResponse:\n{output_str}"""

    return prompt

def format_inference_prompt(combined_query_string):
    """Formats prompt, separating instruction and context, adding emphasis."""
    instruction_part = combined_query_string
    input_part = None
    context_marker_match = re.search(r'Context\s*[:\-]?\s*', combined_query_string, re.IGNORECASE)
    if context_marker_match:
        marker_end_index = context_marker_match.end()
        instruction_part = combined_query_string[:context_marker_match.start()].strip()
        input_part = combined_query_string[marker_end_index:].strip()
        if not input_part:
            input_part = None
            instruction_part = combined_query_string

    instruction_for_prompt = instruction_part
    if input_part:
        instruction_for_prompt = f"{instruction_part}\n\nIMPORTANT: Use the provided context below (in the 'Input' section) to determine the answer."

    prompt = create_prompt_base(instruction_for_prompt, input_part, output="", include_eos=False)

    response_marker = "\nResponse:"
    if prompt.endswith(response_marker):
        prompt = prompt[:-len(response_marker)].rstrip()
    return prompt

def select_best_by_rouge(candidate_texts, rouge_calculator):
    """Selects the best candidate text using ROUGE L F1 score."""
    if not candidate_texts:
        return "[NO VALID ANSWERS]"
    if len(candidate_texts) == 1:
        return candidate_texts[0]

    avg_rouge_l_f1_scores = []
    final_answer = "[ROUGE FALLBACK]"
    try:
        for c_idx, current_candidate in enumerate(candidate_texts):
            other_candidates = [candidate_texts[j] for j in range(len(candidate_texts)) if c_idx != j]
            if not other_candidates:
                avg_rouge_l_f1_scores.append(0)
                continue
            scores = rouge_calculator.compute(
                predictions=[current_candidate] * len(other_candidates),
                references=other_candidates,
                use_stemmer=True,
                rouge_types=['rougeL']
            )
            avg_score = scores.get('rougeL', 0.0)
            avg_rouge_l_f1_scores.append(avg_score)

        if avg_rouge_l_f1_scores:
            best_candidate_index = np.argmax(avg_rouge_l_f1_scores)
            if 0 <= best_candidate_index < len(candidate_texts):
                final_answer = candidate_texts[best_candidate_index]
            else:
                print(" >> Warning: Invalid best index in ROUGE.")
                final_answer = candidate_texts[0]
        else:
            final_answer = candidate_texts[0] if candidate_texts else "[ROUGE CALC ERROR]"
    except Exception as e_sim:
        print(f"  >> Error during ROUGE calc: {e_sim}")
        final_answer = candidate_texts[0] if candidate_texts else "[ROUGE ERROR FALLBACK]"

    return final_answer

# FastAPI App Setup
app = FastAPI(title="LLM Assistant API (vLLM + ROUGE SC)", description="API using vLLM engine with ROUGE Self-Consistency.")

# Startup Event: Load Model & Resources
@app.on_event("startup")
async def startup_event():
    global llm_engine, tokenizer_for_prompting, rouge_metric, create_prompt_func
    print("Server starting up...")
    if llm_engine is not None:
        print("vLLM Engine already loaded.")
        return

    # 1. Load Tokenizer (needed for prompt formatting)
    print("Loading Tokenizer...")
    try:
        tokenizer_for_prompting = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
        if tokenizer_for_prompting.pad_token is None:
            tokenizer_for_prompting.pad_token = tokenizer_for_prompting.eos_token
        tokenizer_for_prompting.padding_side = "left"
        print("Tokenizer loaded for prompt formatting.")
    except Exception as e:
        print(f"FATAL: Failed to load tokenizer from {MERGED_MODEL_PATH}: {e}")
        raise RuntimeError(f"Tokenizer load failed: {e}") from e

    create_prompt_func = create_prompt_base

    # 2. Load ROUGE Metric
    print("Loading ROUGE metric...")
    rouge_metric = evaluate.load('rouge')

    # 3. Load vLLM Engine
    print(f"Initializing vLLM engine from: {MERGED_MODEL_PATH}")
    try:
        llm_engine = LLM(
            model=MERGED_MODEL_PATH,
            tokenizer=MERGED_MODEL_PATH,
            trust_remote_code=True,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )
        print("vLLM Engine loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load vLLM engine: {e}")
        import traceback; print(traceback.format_exc())
        raise RuntimeError(f"vLLM Engine failed to load: {e}") from e

# Shutdown Event: Clean up Resources
@app.on_event("shutdown")
async def shutdown_event():
    global llm_engine, tokenizer_for_prompting, rouge_metric
    print("Server shutting down...")
    if llm_engine is not None: del llm_engine
    if tokenizer_for_prompting is not None: del tokenizer_for_prompting
    if rouge_metric is not None: del rouge_metric
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("Resources released.")

# Generation Endpoint
@app.post("/generate", summary="Generate text using vLLM + ROUGE SC")
async def generate(request: QueryRequest):
    """
    Receives user_query, formats prompt, generates candidates via vLLM,
    selects best via ROUGE, returns result.
    """
    global llm_engine, rouge_metric, tokenizer_for_prompting

    if llm_engine is None or rouge_metric is None or tokenizer_for_prompting is None:
        raise HTTPException(status_code=503, detail="Server resources not ready.")

    try:
        start_time = time.time()

        # 1. Format the input prompt
        formatted_prompt = format_inference_prompt(request.user_query)

        # 2. Define Sampling Parameters for vLLM
        sampling_params = SamplingParams(
            n=DEFAULT_N_SAMPLES,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            max_tokens=DEFAULT_MAX_NEW_TOKENS,
        )

        # 3. Generate Candidates with vLLM
        vllm_outputs = llm_engine.generate([formatted_prompt], sampling_params, use_tqdm=True)
        # 4. Extract and Clean Candidate Texts
        if not vllm_outputs or not vllm_outputs[0].outputs:
            print(f"Warning: vLLM returned no outputs for prompt: {formatted_prompt[:100]}...")
            final_answer = "[VLLM GENERATION FAILED]"
        else:
            candidate_texts = []
            response_marker_to_remove = "Response:"
            for completion in vllm_outputs[0].outputs:
                cleaned_text = completion.text.strip()
                if cleaned_text.startswith(response_marker_to_remove):
                    cleaned_text = cleaned_text[len(response_marker_to_remove):].lstrip()
                if cleaned_text:
                    candidate_texts.append(cleaned_text)
            print("-" * 20 + " Generated Candidates " + "-" * 20)
            for i, cand_text in enumerate(candidate_texts):
                print(f"Candidate {i+1}: {cand_text[:150]}...") 
            print("-" * 58)
            # 5. Select Best Candidate using ROUGE
            final_answer = select_best_by_rouge(candidate_texts, rouge_metric)

        end_time = time.time()
        print(f"Processed request in {end_time - start_time:.2f}s. Input: '{request.user_query[:50]}...' Output: '{final_answer[:50]}...'")

        return {"response": final_answer}

    except Exception as e:
        print(f"ERROR during generation request: {e}")
        import traceback; print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process request: {e}")

# Root Endpoint
@app.get("/", summary="Root endpoint for basic check")
async def read_root():
    """Basic health check endpoint."""
    return {"message": "LLM Assistant API (vLLM) is running."}

# Run the server
if __name__ == "__main__":
    print("Starting FastAPI server with uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8004)
