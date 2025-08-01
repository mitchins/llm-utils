import re
from enum import Enum
import json

class LLMResponse(Enum):
    AFFIRMATIVE = "AFFIRMATIVE"
    NEGATIVE = "NEGATIVE"
    FAILED = "FAILED"

def process_llm_response(response, console=None):
    lines = response.strip().split("\n")

    for line in lines:
        cleaned = re.sub(r"^(?:\*\*+)?\s*Answer\s*[:：]\s*", "", line.strip(), flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\[\s*y\s*/\s*n\s*\]\s*[:：]?", "", cleaned, flags=re.IGNORECASE).strip()

        token = re.sub(r'\s+', '', cleaned.lower())
        match = re.match(r"^[\[\(\*'\"#]*([yn])(?!/)[\]\)\*'\"#.:]*", token)
        if not match:
            # Fallback: scrub line of known prompt pattern and re-evaluate
            fallback_cleaned = re.sub(r"^.*?\[\s*y\s*/?\s*n\s*\]\s*[:：]?\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
            fallback_cleaned = re.sub(r"^.*?\*\*[yn]\*\*", lambda m: m.group(0)[-4:], fallback_cleaned, flags=re.IGNORECASE)
            token = re.sub(r'\s+', '', fallback_cleaned.lower())
            match = re.match(r"^\w+ ?[\[\(\*'\"#]*([yn])[\]\)\*'\"#.:]*", token)

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

# TODO: Write tests independent of json parsing
def scrub_think(text: str) -> str:
    """
    Remove any <think>...</think> or <reasoning>...</reasoning> tags,
    including HTML-escaped forms.
    """
    # Remove HTML-escaped tags
    cleaned = re.sub(r"&lt;/?(?:think|reasoning)&gt;", "", text, flags=re.IGNORECASE)
    # Remove explicit tags
    cleaned = re.sub(r"<(?:think|reasoning)>.*?</(?:think|reasoning)>", "", cleaned,
                     flags=re.DOTALL | re.IGNORECASE)
    return cleaned

def extract_json_structure(text):
    """
    Extract JSON content from markdown-style code block or the first outermost JSON array or object using regex.
    Supports both JSON arrays and dictionaries.
    """
    text = text.strip()

    # Remove <think>...</think> or <reasoning>...</reasoning> sections if present
    text = scrub_think(text)

    # Try to parse the entire text as JSON first
    try:
        parsed = json.loads(text)
        # Return a canonical JSON string (stripping extraneous whitespace)
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        pass

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

def extract_and_parse(text: str):
    """
    Extract embedded JSON from text and parse it into Python objects.
    """
    cleaned = extract_json_structure(text)
    return json.loads(cleaned)
