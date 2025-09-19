import time
import openai

def call_llm_with_retry(llm_client, max_retries=3, **kwargs):
    """Call LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            return llm_client.chat.completions.create(extra_body={}, **kwargs)
        except openai.InternalServerError as e:
            if "504" in str(e) and attempt < max_retries - 1:
                wait_time = 2**attempt
                print(f"Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                raise
        except Exception as e:
            print(f"Error calling LLM: {e}")
            raise
    raise RuntimeError(f"Failed after {max_retries} attempts")