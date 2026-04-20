def get_value_prompt(key, expected_output, ocr_context):
    return f"""
You are a strict product-attribute value extractor.

Your task:
- Extract ONLY concrete values from OCR-extracted text that are VALID according to the attribute rules defined below, and that could legitimately be assigned as the value for the product attribute "{key}".

ABSOLUTE RULES (DO NOT VIOLATE):
1. Output MUST be a valid Python list literal.
2. Output MUST contain ONLY the list — no explanations, comments, labels, or extra text.
3. Do NOT output JSON, dictionaries, keys, attribute names, or booleans unless they are the value itself.
4. Each list element MUST be a single value that could directly be assigned to the key "{key}".
5. Values may appear in:
   - table cells
   - column values
   - label–value pairs
   - short descriptive text
6. If a value appears in a table, return ONLY the cell value itself — NOT headers, row labels, footnotes, or surrounding text.
7. EXCLUDE any value that is:
   - an instruction, note, or reference (e.g. “See page 2”, “Refer to brochure”)
   - a SKU, part number, model number, or internal code
   - a placeholder or capability (e.g. “Custom”, “Various”, “Available upon request”)
   - marketing or descriptive text that does NOT represent an actual value
8. Do NOT hallucinate, infer missing values, or combine multiple fields.
9. Preserve original wording and formatting as found in the OCR text, except for trimming whitespace.
10. If no valid value exists, return an empty Python list: [].
11. You MUST stop extracting once all DISTINCT valid values present in the OCR text
    have been identified.
12. Do NOT continue generating values beyond what is explicitly present in the OCR text.
13. The output list MUST be the minimal complete set of valid values.
    Adding extra values is a violation.

ATTRIBUTE VALUE RULES (AUTHORITATIVE):
The following describes what constitutes a VALID value for this attribute.
These rules may allow:
- exact matches to predefined option values
- equivalent variants (spacing, case, minor formatting differences)
- numeric values with specific formatting requirements
- boolean values (true / false) when explicitly stated
- measurements with allowed units or ranges

Apply ONLY the rules that are relevant to the extracted value.
Do NOT return values that violate these rules.

{expected_output}

Attribute key (REFERENCE ONLY — DO NOT OUTPUT):
{key}

OCR text (unstructured, may contain noise):
<<<OCR_START>>>
{ocr_context}
<<<OCR_END>>>

MANDATORY SELF-CHECK (BEFORE OUTPUT):
For each candidate value, confirm ALL of the following:
1. Is this a real, concrete value (not a note, reference, or code)?
2. Does it satisfy at least one of the allowed formats or conditions described above?
3. Could this value alone be stored as the value for "{key}" in a database or shown as a selectable option?

If ANY answer is NO, the value MUST be excluded.
NEVER INCLUDE DESCRIPTIONS VALUE CAN'T BE LONGER THAN 3 WORDS
OUTPUT FORMAT REQUIREMENTS:
- Output must be parseable by Python ast.literal_eval
- Use double quotes for string values

REMEMBER:
OUTPUT A PYTHON LIST AND NOTHING ELSE.
"""

def build_ocr_context(ocr_key_hit, key):
    """
    Builds a single string of context for a specific key.
    Filters through hit_objects and joins their context snippets.
    """
    # 1. Use a list comprehension to find all contexts where the key matches
    # We use .lower() on both to ensure the match isn't broken by capitalization
    context_snippets = [
        hit_object['context']
        for hit_object in ocr_key_hit
        if hit_object.get('key', '').lower() == key.lower()
    ]

    # 2. Join the snippets with a single space
    # .strip() removes any leading/trailing whitespace from the final result
    return " ".join(context_snippets).strip()

def format_row_data_to_markdown(text_list, key):
    """
    Convert a list of strings into a vertical Markdown table column.

    Args:
        text_list (list): List of strings or dicts containing a 'text' field.
        key (str): The heading for the column.
    """
    if not text_list:
        return f"**Column {key}:**\n> No data to display.\n"

    # Start the "Column:" section with the user-provided key
    header = str(key).upper()
    block = [f"**Column:**", f"| {header} |"]

    for item in text_list:
        # Handle if item is a dictionary or just a raw string
        if isinstance(item, dict):
            text = item.get('text', '').strip()
        else:
            text = str(item).strip()

        if text:
            # Add as a new row in the table
            block.append(f"{text} ")

    if len(block) == 3: # Only header and separator exist
        block.append("")

    return "\n".join(block)
