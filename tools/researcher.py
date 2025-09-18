from openai import OpenAI
from dotenv import load_dotenv
import os
import math
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="http://localhost:8000/v1")

def _safe_read(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return ""

# https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a Kaggle Competition query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

def get_relevant_context(threads: list[str], question: str, num_threads=3, threshold=0.75):
    task = 'Given a Kaggle Competition query, retrieve relevant passages that answer the query'
    kb_threads = [_safe_read(p) for p in threads]
    questions = [question] * len(kb_threads)
    pairs = [format_instruction(task, query, doc) for query, doc in zip(questions, kb_threads)]
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)
    ranked_threads = sorted(zip(threads, kb_threads, scores), key=lambda x: x[2], reverse=True)
    # print([[x[0], x[2]] for x in ranked_threads])
    selected_indexes = [t[0] for t in ranked_threads[:num_threads] if t[2] >= threshold]
    print(f"Query: {question}, Selected threads: {selected_indexes}")
    selected_threads = [t[1] for t in ranked_threads[:num_threads] if t[2] >= threshold]
    return "\n\n".join(selected_threads), len(selected_threads)

def ask_domain_expert(question: str):
    slug = os.environ.get("TASK_SLUG", "make-data-count-finding-data-references")
    base_dir = os.path.join("task", slug, "threads")
    threads = [os.path.join(base_dir, p) for p in os.listdir(base_dir) if p.endswith('.md')]
    strategic_brief, total_cited_threads = get_relevant_context(threads, question)

    if total_cited_threads == 0:
        return "Your question cannot be answered based on the competition discussion threads."

    prompt = f"""You are an Kaggle Grandmaster who is a domain expert in the given competition. You are given a question raised by a competition participant, as well as up till 3 strategic briefs of relevant discussion threads, and you should answer it to the best of your ability. If you do not know the answer, respond with 'Your question cannot be answered based on the competition discussion threads.'.

Strategic Briefs:
{strategic_brief}

Question:
{question}
    """

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = client.chat.completions.create(
        model="bogoconic1/Qwen3-4B-Instruct-MDC-Discussion-SFT",
        messages=messages
    )
    return response.choices[0].message.content
    

def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "ask_domain_expert",
                "description": "Ask a question to the domain expert",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to ask the domain expert"}
                    },
                    "required": ["question"],
                },
            },
        }
    ]