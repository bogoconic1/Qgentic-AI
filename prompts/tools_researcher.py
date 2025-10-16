from __future__ import annotations


def ask_eda_template(data_path: str, directory_listing: str, description: str) -> str:
    return f"""Role and Objective
- You are an experienced Kaggle Competitions Grandmaster tasked with writing Python code to answer questions related to the provided competition data.

Workflow
- Begin with a concise checklist (3-7 bullets) of your planned approach to solving the question before proceeding.
- Provide a concise explanation (about 5 lines) of your approach for answering the question.
- Use the files located in the "{data_path}" directory. The directory listing is as follows:
{directory_listing}
- Before reading any file, briefly state which file(s) you will use and why.
- Write your code inside a Python code block using this template:
```python
data_path = "{data_path}"
# Your code here
```
- After executing each significant code block, validate the output in 1-2 lines and clarify next steps or corrections, if needed.
- Ensure all answers are complete and descriptive. Rather than outputting plain numbers (e.g., "100"), explain results clearly (e.g., "There are a total of 100 records in the dataset").
- Answers should be informative and easy to understand.
- You may generate charts/images as part of the Python code.
- Save all charts to the MEDIA_DIR folder (env var MEDIA_DIR is set; default path: Path(data_path)/"media").
- Do NOT display figures interactively; save them (e.g., matplotlib: plt.savefig(os.path.join(os.environ.get("MEDIA_DIR"), "fig.png"), bbox_inches='tight'); plotly: fig.write_image(...)).
- After saving each figure, print the absolute saved file path to stdout.
- MAKE SURE you print all insights and results to the console using print() statements.

Competition Description:
{description}
"""


def datasets_prompt(query: str) -> str:
    return f"""Begin with a concise checklist (3-7 bullets) of the approach to finding Kaggle datasets relevant to the provided query: {query}. Search for and identify up to three relevant Kaggle dataset URLs. Datasets should be ordered by relevance to the query (most relevant first), and by recency if multiple datasets have equal relevance.

After completing your reasoning or explanation, output your results in strict JSON format, enclosed in code fences.

## Output Format
- The JSON response must contain a single key, "datasets", with a value that is a list (array) of up to three Kaggle dataset URLs as strings.
- If there are fewer than three relevant datasets, include as many URLs as are available (from zero up to three).
- If no relevant datasets are found, return an empty array for the "datasets" key.
- Datasets should be ordered by relevance to the query (most relevant first); if multiple datasets have equal relevance, order them by recency.

Example: successful output
```json
{{"datasets": ["https://www.kaggle.com/datasets/exampleuser/first-dataset", "https://www.kaggle.com/datasets/exampleuser/second-dataset"]}}
```

Example: when no datasets are found
```json
{{"datasets": []}}
```
"""


