import logging
import re
import json
from  input_handler import load_attribute_schema
from  model_loader import get_llm_instance

def build_mounting_prompt(product_type_set):
    prompt = f'''
You are a lighting product domain expert.
You are given a list of lighting product type names:
{product_type_set}

For each product type, return a JSON array of 1-4 identifying keywords.
These keywords must be terms that would literally appear in a manufacturer spec sheet for that product, and that distinguish it from other product categories.

Apply these rules based on what kind of name it is:

RULE 1 — Abbreviation / certification / compliance / program labels
  (names that are acronyms, certifications, compliance programs, or industry labels)
  → Return only the abbreviation(s) or the exact label phrase. Do NOT add generic descriptors.
  Pattern: ["<ACRONYM>"] or ["<label phrase>"]

RULE 2 — Physical fixture or application type
  (names describing a physical product or application)
  → Return the mounting style and/or the most specific product descriptor or alias.
  Pattern: ["<mounting>", "<descriptor>", "<alias if any>"]

RULE 3 — Avoid these generic terms entirely — they appear in all spec sheets and provide no discrimination:
  "surface", "light", "lamp", "LED", "luminaire", "fixture", "indoor", "outdoor", "unit", "system", "product"

RULE 4 — Case:
  Lowercase for physical/mounting terms.
  Preserve standard case for abbreviations and acronyms.

RULE 5 — Compound product names that use "and", "&", or "/" may describe two sub-categories.
  Include aliases for each sub-category so either can trigger a match.

Each key must be the exact original product type string from the input list.
Each value must be a JSON array of strings — never a plain string, never an empty array.
Return ONLY valid JSON. No markdown, no comments, no explanations.
/no_think
'''
    return prompt


_DEFAULT_JSON_SYSTEM_PROMPT = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are a JSON-specialized assistant with strict formatting rules. Do not output markdown or natural language. Do not overthink or explain your steps. Think carefully and analyze the input thoroughly before responding. Your output MUST:

            1. Be valid JSON (passes JSONLint)
            2. Quote all keys with double quotes
            3. Contain no duplicate keys
            4. Use arrays with uniform types
            5. Use proper escaping for special characters
            7. Never include markdown, commentary, or explanations

            Output only the JSON object. Nothing else.
            \no_think
            """

def generate_llm_response(prompt, use_gpu=False, system_prompt=_DEFAULT_JSON_SYSTEM_PROMPT):
    llm = get_llm_instance(use_gpu)
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    # print(f"[DEBUG][generate_llm_response] system_prompt={'<none>' if system_prompt is None else system_prompt[:80] + '...'}")
    # print(f"[DEBUG][generate_llm_response] sending {len(messages)} message(s) to llama_cpp")
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.9,
        max_tokens=4096,
        top_p=0.9,
    )
    content = response["choices"][0]["message"]['content']
    # print(f"[DEBUG][generate_llm_response] raw llama_cpp content:\n{content}")
    return content

def load_schema_and_derive_product_types(schema_path):
    logging.info(f"Loading schema from {schema_path}...")
    schema = load_attribute_schema(schema_path)
    product_type_set = set()
    for key in schema:
        product_type_set.update(schema[key]["product_types"])
    logging.info(f"Derived {len(product_type_set)} unique product types.")
    return schema, product_type_set


def fix_and_load_json(response_text):
    logging.info("Attempting to parse and sanitize JSON response...")
    response_text = response_text.strip()

    # Extract JSON block if wrapped in text
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        response_text = match.group(0)

    # Remove trailing commas ONLY
    response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)

    try:
        parsed = json.loads(response_text)
        logging.debug("JSON successfully parsed.")
        return parsed
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse JSON: {e}")
        raise ValueError(f"Invalid JSON: {e}")

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def remove_think_block(text: str) -> str:
    if not text:
        return text
    return THINK_BLOCK_RE.sub("", text).strip()

def get_valid_json(prompt, initial_response=None, max_retries=3, use_gpu=False):
    """
    Validates an initial LLM response and regenerates only if necessary.
    Automatically removes <think>...</think> blocks before parsing.
    """

    if initial_response is not None:
        try:
            cleaned = remove_think_block(initial_response)
            return fix_and_load_json(cleaned)
        except ValueError:
            logging.info("Initial response is invalid JSON; regenerating...")

    for attempt in range(max_retries):
        logging.info(f"LLM JSON generation attempt {attempt + 1}/{max_retries}")
        response = generate_llm_response(prompt, use_gpu)

        cleaned = remove_think_block(response)

        try:
            return fix_and_load_json(cleaned)
        except ValueError:
            logging.warning(
                f"Attempt {attempt + 1} failed: invalid JSON. Retrying..."
            )

    raise RuntimeError("Failed to get valid JSON after multiple attempts")

