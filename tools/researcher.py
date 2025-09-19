from openai import OpenAI
from dotenv import load_dotenv
import os
import math
from typing import List
import sys
import re
from io import StringIO
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

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
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()

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
        extra_body={},
        model="qwen/qwen3-next-80b-a3b-instruct",
        messages=messages
    )
    return response.choices[0].message.content


def ask_eda(question: str, data_description: str, overview: str, data_path: str, max_attempts: int = 5) -> str:
    """Asks a question about the data provided for exploratory data analysis (EDA)"""
    PROMPT = f"""You are an experienced Kaggle Competitions Grandmaster. Your goal is to write code that answers questions about the data provided.
Competition Overview:
{overview}

Data Description:
{data_description}

You will be given one or more questions related to the data. Your task is to generate code that, when executed, will answer all questions using the data provided.
Before generating the code, provide around 5 lines of reasoning about the approach.

The data files are stored in the directory "{data_path}".

So make sure you put your code within a python code block like this:
```python
data_path = "{data_path}"
<YOUR CODE HERE>
```

IMPORTANT: Always provide descriptive answers. Instead of just printing a number like "100", print a complete sentence like "There are a total of 100 records". Make your final answer clear and informative.
IMPORTANT: If you read in train_labels.csv, make sure to remove records with type = 'Missing' before doing any analysis.
"""
    all_messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": question},
    ]

    pattern = r'```python\s*(.*?)\s*```'

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        print(f"Attempt {attempt} to answer the question.")
        completion = client.chat.completions.create(
            extra_body={},
            model="qwen/qwen3-coder",
            messages=all_messages
        )
        response_text = completion.choices[0].message.content or ""

        matches = re.findall(pattern, response_text, re.DOTALL)
        code = "\n\n".join(matches).strip()
        print(code)

        if not code:
            last_error = "No python code block found in the model response."
        else:
            try:
                # Simple policy assertion carried forward
                assert "train_labels.csv" not in code or "!= 'Missing'" in code, (
                    "You must remove records with type = 'Missing' before doing any analysis."
                )

                # Save current stdout
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                exec(code)
                output = captured_output.getvalue()

                sys.stdout = old_stdout

                return output

            except AssertionError as e:
                try:
                    sys.stdout = old_stdout
                except Exception:
                    pass
                last_error = f"Assertion failed: {e}"
                print(last_error)
            except Exception as e:
                try:
                    sys.stdout = old_stdout
                except Exception:
                    pass
                last_error = f"Error executing code: {str(e)}"
                print(last_error)

    return "Your question cannot be answered."
    

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
        },
        {
            "type": "function",
            "function": {
                "name": "ask_eda",
                "description": "Ask a question to the EDA expert",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to ask the EDA expert"}
                    },
                    "required": ["question"],
                },
            },
        }
    ]