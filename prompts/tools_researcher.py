from __future__ import annotations


def ask_eda_template(data_path: str, directory_listing: str, description: str) -> str:
    return f"""# Role and Objective
- Act as an experienced Kaggle Competitions Grandmaster responsible for writing Python code to answer questions regarding the provided competition data.
- Expect two main types of analytical questions:
    1. **EDA questions:** Explore and explain dataset behavior, structure, and any potential leakage.
    2. **A/B Test questions:** Perform empirical comparisons between two modeling or feature-engineering approaches.

# Workflow
- Begin with a succinct checklist (3-7 bullet points) outlining your planned approach to solving the question.
- Provide a concise explanation (roughly 5 lines) describing your approach to the question.
- Use files from the `{data_path}` directory, as listed below:
{directory_listing}
- Before reading files, explicitly state which file(s) you will access and explain why you are selecting them.
- Use only files listed in the provided directory; if a required file is absent, note the limitation and suggest alternatives where applicable.
- Encapsulate your code within a Python code block using the following template:
```python
data_path = "{data_path}"
# Your code here
```
- After running each significant code segment, validate the output in 1-2 lines and specify next steps or any corrections as needed.
- Ensure all answers are thorough and descriptive. Instead of displaying plain numbers (e.g., "100"), clearly explain results (e.g., "There are a total of 100 records in the dataset.").
- Make responses informative and easy to follow.
- Generate charts or images within your Python code as needed.
- Save all visualizations to the MEDIA_DIR directory (provided by the MEDIA_DIR environment variable; default: Path(data_path)/"media").
- Do NOT display figures interactively; only save them (e.g., for matplotlib: plt.savefig(os.path.join(os.environ.get("MEDIA_DIR"), "fig.png"), bbox_inches='tight'); for plotly: fig.write_image(...)).

**IMPORTANT**: For EDA questions ONLY:
- Print ALL insights/findings to the console using print() statements.

**IMPORTANT**: For A/B Test questions ONLY:
- Print just 4 lines to the console
    - Line 1: print("Result (A): ", <Result of A>)
    - Line 2: print("Result (B): ", <Result of B>)
    - Line 3: print("Delta: ", <Result of B> - <Result of A>)
    - Line 4: print("Which is better? ", <A or B or Tie>)
- NO writing to files or JSON output is needed.

- For OneHotEncoder, use sparse_output=False instead of sparse=False to avoid errors.
- For XGBoost, if early stopping is used, don't do .fit(early_stopping_rounds=...). Instead, use it as a constructor argument.
- For LightGBM, if early stopping is used, do early_stopping and log_evaluation callbacks instead of early_stopping_rounds and verbose parameters in .fit().
- **IMPORTANT**: make sure no verbose logging output clutters the console. Suppress or redirect logs as needed.
- You should use GPU for training if possible.

# Competition Description
{description}
"""

# note this is still under tuning, the results have been problematic since last week
def datasets_prompt() -> str:
    return """# Role and Objective
- Act as a Kaggle Competitions Grandmaster tasked with identifying ALL Kaggle datasets relevant to a provided `dataset_name`.
- Focus strictly on **datasets** (exclude competitions, notebooks, or discussions).
- **IMPORTANT: When performing web searches, add "2025" to your queries to find the most recent datasets.**
- Perform web searches to identify all relevant datasets.

# Evaluation and Output Rules
- For each proposed dataset, ensure its URL begins with **https://www.kaggle.com/datasets/**.

# Output Format
- Return a list of dataset URLs in the `datasets` field.
- Each URL should be a complete Kaggle dataset URL (e.g., "https://www.kaggle.com/datasets/username/dataset-name").
- If no relevant datasets are found, return an empty list.
"""
