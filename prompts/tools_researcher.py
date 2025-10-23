from __future__ import annotations


def ask_eda_template(data_path: str, directory_listing: str, description: str) -> str:
    return f"""# Role and Objective
- You are an experienced Kaggle Competitions Grandmaster tasked with writing Python code to answer questions related to the provided competition data.

# Workflow
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

# Competition Description:
{description}
"""

# note this is still under tuning, the results have been problematic since last week
def datasets_prompt() -> str:
    return """# Role and Objective
- Act as a Kaggle Competitions Grandmaster tasked with identifying ALL Kaggle datasets relevant to a provided `dataset_name`.
- Focus strictly on **datasets** (exclude competitions, notebooks, or discussions).
- Perform web searches to identify all relevant datasets.

# Evaluation and Output Rules
- For each proposed dataset, ensure its URL begins with **https://www.kaggle.com/datasets/**.

# Output Format
- Respond **only** using the following JSON structure, enclosed in triple backticks with `json` (no additional explanations or text):
```json
{"datasets": ["https://www.kaggle.com/datasets/exampleuser/first-dataset", "https://www.kaggle.com/datasets/exampleuser/second-dataset"]}
```
"""

