import re
from difflib import SequenceMatcher

def build_regex_prompt(input):
    Regex_prompt = f'''
### STRICT INSTRUCTIONS
You are a JSON-only regex generator. Follow these rules EXACTLY:

1. **OUTPUT FORMAT IS SACRED:**
   - Output MUST be a valid JSON object with ONLY top-level keys from the input.
   - Each key maps to an object with EXACTLY ONE field: `"pair_regex"`.
   - `"pair_regex"` is either:
        • A STRING containing a properly escaped regex pattern (e.g., `"\\\\bKey\\\\b"`), OR
        • `null` (literal JSON null, NOT string "null")
   - NO additional fields, comments, or text outside JSON.

2. **NULL RULES (RETURN `null` IF):**
   - `"Expected Output Formatting"` contains ANY of these phrases (case-insensitive):
     `"exactly match"`, `"predefined list"`, `"must be one of"`, `"strictly match"`,
     `"true/false statement"`, `"enum"`, `"categorical"`, `"fixed options"`
   - Values are static strings without numeric/percent patterns (even with case variations)
   - No dynamic elements exist in values (digits, %, units, wildcards)
   - ⚠️ **Exception**: If the formatting says `"Return Any Values"` or `"Return All Values"` (case-insensitive), **DO NOT apply NULL RULE** — proceed to REGEX RULE instead, **unless** a null-trigger phrase also appears (then NULL wins).

3. **REGEX RULES (ONLY WHEN NULL RULES DON'T APPLY):**
   - Pattern must match: `[CASE-INSENSITIVE KEY] + [SEPARATOR] + [VALUE PATTERN]`
   - Mandatory prefix: `(?i)` for case insensitivity
   - Key must be wrapped in `\\b` word boundaries: `(?i)\\bKey Name\\b`
   - Separator pattern: `[\\s:-]+` (matches spaces/colons/dashes)

   - **VALUE PATTERNS depend on context:**
        a) **If "Return Any Values" or "Return All Values" is present** → match **exactly one word** (a single non-whitespace token). Use: `\\\\S+`
           → Full pattern: `(?i)\\\\b{{KEY}}\\\\b[\\\\s:-]+\\\\S+`

        b) **For Numeric measurements** (e.g., 4in, 2x2): `\\\\d+(?:\\\\.\\\\d+)?(?:\\\\s*[a-zA-Z%]+)?`
        c) **For Watts**: `\\\\d+(?:\\\\.\\\\d+)?\\\\s*[wW]`
        d) **For Percentages**: `\\\\d+(?:\\\\.\\\\d+)?%`
        e) **NEVER hardcode values from "values" array**

   - Escape ALL regex metacharacters properly (e.g., `.` → `\\.`)
   - In the one-word case, `\\\\S+` ensures no spaces — matching tokens like "black", "100W", "3.5", "N/A"

4. **ANTI-FAILURE MEASURES:**
   - Test your regex mentally: Would `"Color: red"` match? Yes. `"Color: red blue"`? Only "red" would be considered valid; full match fails after space — which is correct.
   - If uncertain between null/regex → DEFAULT TO `null`
   - DOUBLE ESCAPE backslashes: `\b` → `\\\\b`, `\s` → `\\\\s`, `\S` → `\\\\S`
   - NO trailing commas, comments, or markdown formatting

**EXAMPLE VALID OUTPUTS:**
✅ Correct (one word – "Return All Values"):
{{
  "Finish": {{
    "pair_regex": "(?i)\\\\bFinish\\\\b[\\\\s:-]+\\\\S+"
  }}
}}

✅ Correct (numeric):
{{
  "Input Watts": {{
    "pair_regex": "(?i)\\\\bInput Watts\\\\b[\\\\s:-]+\\\\d+(?:\\\\.\\\\d+)?\\\\s*[wW]"
  }}
}}

✅ Correct (null case):
{{
  "Status": {{
    "pair_regex": null
  }}
}}

❌ INVALID OUTPUTS (CAUSE SYSTEM FAILURE):
• Any text before/after JSON
• Single backslashes: `"\\bKey\\b"`
• String `"null"` instead of literal null
• Extra fields like `"notes": "..."`
• Using `.*` or `.+` in wildcard case (must be `\\\\S+` for one word)

### INPUT DATA
{input}

### OUTPUT (VALID JSON ONLY):
'''
    return Regex_prompt


def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\(.*?\)', '', text)   # remove parentheticals
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def prefix_key(text, prefix_len=4):
    """
    Returns the first `prefix_len` tokens as a grouping key.
    """
    tokens = text.split()
    return " ".join(tokens[:prefix_len])


def sentence_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def group_schema_by_sentence_closeness(normalized_schema, threshold=0.5):
    """
    Groups attributes by how close their formatting sentences are.
    """

    items = []
    for name, details in normalized_schema.items():
        instr = details.get("Expected Output Formatting", "")
        items.append({
            "name": name,
            "details": details,
            "text": normalize_text(instr)
        })

    groups = []
    visited = set()

    for i, base in enumerate(items):
        if i in visited:
            continue

        group = {base["name"]: base["details"]}
        visited.add(i)

        for j in range(i + 1, len(items)):
            if j in visited:
                continue

            sim = sentence_similarity(base["text"], items[j]["text"])
            if sim >= threshold:
                group[items[j]["name"]] = items[j]["details"]
                visited.add(j)

        groups.append(group)

    return groups

    return list(groups.values())



def clean_guidance(guidance_list):
    # Keys we want to remove
    keys_to_remove = {'data_type', 'product_types'}

    return [
        {
            key: {
                k: v for k, v in value.items() if k not in keys_to_remove
            }
            for key, value in item.items()
        }
        for item in guidance_list
    ]

# Usage:
# cleaned_guidance = clean_guidance(guidance)