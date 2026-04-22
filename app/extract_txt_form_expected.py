def get_value_prompt(key, expected_output, ocr_context):
    return f"""
You are a product-attribute value extractor. Your only job is to find values for a single attribute from OCR text.

ATTRIBUTE: {key}

--- EXTRACTION RULE (READ THIS FIRST - IT CONTROLS EVERYTHING) ---
{expected_output}
---

This rule tells you:
- What values are valid (exact list match, similar variants, or free-form numeric)
- What format returned values must follow (units, casing, numeric style)
- How strictly to filter what you extract

Apply this rule literally.
If it says 'must exactly match', only return values that directly correspond to known options for this attribute.
If it says 'return all values' or gives a numeric format, extract every occurrence that fits that format.
If it says 'may return similar', use the rule as a guide but allow reasonable variants found in the text.

--- UNIVERSAL EXCLUSIONS (always apply regardless of rule above) ---
- Notes, references, instructions (e.g. 'See page 2', 'Refer to brochure')
- SKUs, part numbers, model codes
- Placeholders ('Custom', 'Various', 'Available upon request')
- Marketing text that is not a concrete assignable value
- Table headers, row labels, footnotes
- Values longer than 3 words unless the extraction rule explicitly allows it

--- OUTPUT FORMAT ---
- Output a valid Python list literal and NOTHING ELSE
- Use double quotes for string values
- If nothing valid found: []
- Do NOT hallucinate or infer - only extract what is explicitly present in the OCR text

OCR TEXT:
<<<OCR_START>>>
{ocr_context}
<<<OCR_END>>>

SELF-CHECK BEFORE OUTPUT: For each candidate value, ask - does it satisfy the EXTRACTION RULE above?
If no, exclude it.

OUTPUT A PYTHON LIST AND NOTHING ELSE.
"""

def build_ocr_context(ocr_key_hit, key):
    """
    Builds a single string of context for a specific key.
    Filters through hit_objects and joins their context snippets.
    """
    context_snippets = [
        hit_object['context']
        for hit_object in ocr_key_hit
        if hit_object.get('key', '').lower() == key.lower()
    ]

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

    header = str(key).upper()
    block = [f"**Column:**", f"| {header} |"]

    for item in text_list:
        if isinstance(item, dict):
            text = item.get('text', '').strip()
        else:
            text = str(item).strip()

        if text:
            block.append(f"{text} ")

    if len(block) == 3:
        block.append("")

    return "\n".join(block)
