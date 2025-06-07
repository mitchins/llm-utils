import re
from enum import Enum

class LLMResponse(Enum):
    AFFIRMATIVE = "AFFIRMATIVE"
    NEGATIVE = "NEGATIVE"
    FAILED = "FAILED"

def process_llm_response(response, console=None):
    lines = response.strip().split("\n")

    for line in lines:
        cleaned = re.sub(r"^(?:\*\*+)?\s*Answer\s*[:：]\s*", "", line.strip(), flags=re.IGNORECASE).strip()
        token = re.sub(r'\s+', '', cleaned.lower())
        match = re.match(r"^[\[*'\"#(]*([yn])[\]*'\"#).:]*", token)
        if match:
            answer = match.group(1).strip().lower()
            if answer in ["y", "yes"]:
                return LLMResponse.AFFIRMATIVE
            elif answer in ["n", "no"]:
                return LLMResponse.NEGATIVE

    if console:
        console.print(f"[bold yellow]Rejected:[/bold yellow] Invalid response format: {response.strip()}", style="bold yellow")
    else:
        print(f"[bold yellow]Rejected:[/bold yellow] Invalid response format: {response.strip()}")
    return LLMResponse.FAILED
    
def extract_json_array(text):
    """
    Extract JSON content from markdown-style code block or the first outermost JSON array/object using regex.
    """
    text = text.strip()

    # Remove <think>...</think> or <reasoning>...</reasoning> sections if present
    text = re.sub(r"&lt;/?(?:think|reasoning)&gt;", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<(?:think|reasoning)>.*?</(?:think|reasoning)>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Try markdown-style fenced block first
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
        return cleaned

    # Try matching a JSON array or object directly using regex
    match = re.search(r"(\[\s*\{.*?\}\s*\]|\{\s*\".*?\"\s*:\s*.*?\})", text, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
        return cleaned

    return text