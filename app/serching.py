import logging
import re
import json
from  helper import generate_ocr_variants
from  input_handler import get_attribute_info_by_key

def matches_key_value_pair(big_text: str, key: str, value) -> bool:
    """
    Check if the big OCR text contains a key-value pair matching the given key and value(s),
    including OCR-tolerant variants.

    Args:
        big_text (str): The concatenated OCR text to search in.
        key (str): The expected key (e.g., "CCT").
        value (str or List[str]): A single value or list of possible values (e.g., ["2700K", "3000K"]).

    Returns:
        bool: True if any value or its OCR variants matches the key in the expected pattern.
    """
    logging.debug(f"Checking for key-value pair: key='{key}', value(s)={value}")
    
    if isinstance(value, str):
        value = [value]

    # Normalize big_text once
    big_text = big_text.lower()

    # Escape and clean the key once
    key_clean = key.strip().lower()
    if not key_clean:
        logging.debug("Key is empty after cleaning – skipping match check.")
        return False
    key_esc = re.escape(key_clean)

    # Common delimiters between key and value
    delimiters = r"[:\-\–\—=]?\s*"

    for val in value:
        val_clean = str(val).strip()
        if not val_clean:
            continue

        # Generate OCR-friendly variants for the value
        ocr_variants = generate_ocr_variants(val_clean)
        logging.debug(f"Generated {len(ocr_variants)} OCR variants for value '{val_clean}': {ocr_variants}")

        for variant in ocr_variants:
            val_esc = re.escape(variant.lower())
            # Pattern: key + optional delimiters + value variant
            pattern = rf"{key_esc}{delimiters}{val_esc}"

            if re.search(pattern, big_text, re.IGNORECASE):
                logging.debug(f"✅ Match found! Pattern '{pattern}' detected in text.")
                return True  # first match is enough
    logging.debug(f"❌ No match found for key '{key}' with any of the provided values or their OCR variants.")
    return False

# def matches_key_value_pair(big_text: str, key: str, value) -> bool:
#     """
#     Check if the big OCR text contains a key-value pair matching the given key and value(s),
#     including OCR-tolerant variants.

#     Args:
#         big_text (str): The concatenated OCR text to search in.
#         key (str): The expected key (e.g., "CCT").
#         value (str or List[str]): A single value or list of possible values (e.g., ["2700K", "3000K"]).

#     Returns:
#         bool: True if any value or its OCR variants matches the key in the expected pattern.
#     """
#     logging.debug(f"Checking for key-value pair: key='{key}', value(s)={value}")

#     if isinstance(value, str):
#         value = [value]

#     # Normalize big_text once
#     big_text_lower = big_text.lower()

#     # Escape and clean the key once
#     key_clean = key.strip().lower()
#     if not key_clean:
#         logging.debug("Key is empty after cleaning – skipping match check.")
#         return False
#     key_esc = re.escape(key_clean)

#     # Common delimiters between key and value
#     delimiters = r"[:\-\–\—=]?\s*"

#     for val in value:
#         val_clean = str(val).strip()
#         if not val_clean:
#             continue

#         # Generate OCR-friendly variants for the value
#         ocr_variants = generate_ocr_variants(val_clean)
#         logging.debug(f"Generated {len(ocr_variants)} OCR variants for value '{val_clean}': {ocr_variants}")

#         for variant in ocr_variants:
#             val_esc = re.escape(variant.lower())
#             pattern = rf"{key_esc}{delimiters}{val_esc}"
            
#             if re.search(pattern, big_text_lower, re.IGNORECASE):
#                 logging.debug(f"✅ Match found! Pattern '{pattern}' detected in text.")
#                 return True

#     logging.debug(f"❌ No match found for key '{key}' with any of the provided values or their OCR variants.")
#     return False

def find_hits(big_text: str, search_terms):
    """
    Searches for occurrences of items in `search_terms` within `big_text`,
    including OCR-tolerant variants.
    Returns a list of original terms that had at least one matching variant in the text.
    """
    big_text_lower = big_text.lower()
    hits = []
    total_terms = len(search_terms)

    logging.debug(f"Searching for {total_terms} term(s) in OCR text (length: {len(big_text)})")

    for term in search_terms:
        term_hits = False
        for variant in generate_ocr_variants(term):
            if variant in big_text_lower:
                hits.append(term)
                term_hits = True
                logging.debug(f"✅ Match found for term '{term}' via variant '{variant}'")
                break  # stop after first successful variant

        if not term_hits:
            logging.debug(f"❌ No match found for term: '{term}'")

    logging.debug(f"Total hits found: {len(hits)} / {total_terms}")
    return hits


# def find_key_hits_from_ocr(keys, ocr_results):
    # """
    # keys: iterable of attribute keys (strings)
    # ocr_results: OCR output list

    # returns: set of original keys that had an exact word hit
    # """
    # logging.debug(f"Searching for {len(keys)} key(s) in OCR results...")

    # # Normalize keys: tuple of words -> original key
    # normalized_key_map = {
    #     tuple(re.findall(r"\b\w+\b", key.lower())): key
    #     for key in keys
    # }
    # matched_keys = set()

    # for result_group in ocr_results:
    #     if not result_group:
    #         continue
    #     ocr_result = result_group[0]
    #     rec_texts = ocr_result.get("rec_texts", [])
    #     for txt in rec_texts:
    #         words = re.findall(r"\b\w+\b", txt.lower())
    #         for key_words, orig_key in normalized_key_map.items():
    #             key_len = len(key_words)
    #             # sliding window exact match
    #             for i in range(len(words) - key_len + 1):
    #                 if tuple(words[i:i + key_len]) == key_words:
    #                     if orig_key not in matched_keys:
    #                         logging.debug(f"✅ Matched key: '{orig_key}' in OCR text: '{txt}'")
    #                     matched_keys.add(orig_key)
    #                     break

    # logging.debug(f"Key search complete. Found {len(matched_keys)} matching key(s): {sorted(matched_keys)}")
    # return matched_keys

_GENERIC_MOUNTINGS = {"surface"}
_NAME_STOP_WORDS = {
    "and", "or", "the", "for", "with", "in", "of", "by", "to",
    "led", "light", "lamp", "lighting", "luminaire", "fixture",
}
# Words that describe a property but aren't distinctive enough to confirm a product type
# on their own — require a non-qualifier token to also match.
_QUALIFIER_WORDS = {
    "compliant", "enabled", "approved", "certified", "rated", "listed",
    "compatible", "integrated", "capable", "ready",
}


def _extract_name_tokens(ptype):
    """
    Returns (content_tokens, qualifier_tokens) for a product type name.

    Content tokens: distinctive words that identify the product category.
    Qualifier tokens: descriptive words ("compliant", "enabled") that are too
    generic to match alone — they only count if a content token also matched.

    Abbreviations (all-caps ≥ 2 chars, like DLC, JA8) and short mixed-case tokens
    (like PoE) are treated as content tokens regardless of length.
    """
    orig_words = re.findall(r"\w+", ptype)
    content, qualifiers = set(), set()
    for orig_w in orig_words:
        w = orig_w.lower()
        if w in _NAME_STOP_WORDS:
            continue
        is_abbrev = orig_w.upper() == orig_w and len(orig_w) >= 2 and orig_w.isalnum()
        is_short_mixed = len(orig_w) >= 2 and any(c.isupper() for c in orig_w) and len(orig_w) <= 5
        if not (len(w) >= 4 or is_abbrev or is_short_mixed):
            continue
        bucket = qualifiers if w in _QUALIFIER_WORDS else content
        bucket.add(w)
        if w not in _QUALIFIER_WORDS and len(w) > 4 and not is_abbrev and not is_short_mixed:
            if w.endswith("es") and len(w) > 5:
                content.add(w[:-2])
            elif w.endswith("s"):
                content.add(w[:-1])
    return sorted(content), sorted(qualifiers)


def _compact(s):
    """Lowercase + strip non-alphanumeric — collapses 'high bay' → 'highbay'."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


_LOOKUP_INDEX_CACHE = {}


def _preprocess_lookup(lookup):
    """
    One-time preprocessing of the product-type lookup.
    Cached by object id so the same in-memory lookup isn't reprocessed
    across multiple PDFs in the same run.
    """
    cache_key = id(lookup)
    cached = _LOOKUP_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    index = {}
    for ptype, val in lookup.items():
        mounting_terms = [val] if isinstance(val, str) else (val if isinstance(val, list) else [])
        specific = [m for m in mounting_terms if m and m.strip().lower() not in _GENERIC_MOUNTINGS]
        content_tokens, qualifier_tokens = _extract_name_tokens(ptype)
        index[ptype] = {
            "content_tokens": content_tokens,
            "qualifier_tokens": qualifier_tokens,
            "mounting_terms": specific,
        }
    _LOOKUP_INDEX_CACHE[cache_key] = index
    return index


def match_product_types_via_lookup(big_text, lookup):
    """
    Detect which product types from the lookup are present in the OCR text.

    Primary: match product type name tokens (incl. singular + compact-text
    fallback so 'Highbay' matches 'high bay' and 'Pendants' matches 'pendant').
    Fallback: match mounting word(s), excluding overly-generic mountings like
    'surface' which appear in virtually every spec sheet.
    """
    logging.debug(f"Matching product types against {len(lookup)} lookup entries...")
    index = _preprocess_lookup(lookup)

    big_text_lower = big_text.lower()
    big_text_compact = _compact(big_text)
    matched = set()

    def _token_in_text(token):
        for variant in generate_ocr_variants(token):
            if variant in big_text_lower or variant in big_text_compact:
                return True
        return False

    for ptype, entry in index.items():
        # --- Primary: name-token match ---
        # A content token must match. If there are no content tokens (edge case),
        # fall back to any qualifier token.
        content_tokens = entry["content_tokens"]
        qualifier_tokens = entry["qualifier_tokens"]

        if content_tokens:
            name_hit = any(_token_in_text(t) for t in content_tokens)
        else:
            # Only qualifiers left — require all of them to reduce false positives
            name_hit = bool(qualifier_tokens) and all(_token_in_text(t) for t in qualifier_tokens)

        if name_hit:
            logging.debug(f"✅ '{ptype}' matched by name.")
            matched.add(ptype)
            continue

        # --- Fallback: specific identifier terms from lookup ---
        if entry["mounting_terms"] and find_hits(big_text, entry["mounting_terms"]):
            logging.debug(f"✅ '{ptype}' matched by identifier fallback: {entry['mounting_terms']}")
            matched.add(ptype)

    logging.debug(f"Matched {len(matched)} product type(s): {sorted(matched)}")
    return matched


def split_schema_by_product_type_match(schema, matched_product_types):
    logging.info(f"Filtering schema by {len(matched_product_types)} matched product type(s)...")
    matched = {}
    not_matched = {}

    for attr_name, attr_obj in schema.items():
        original_product_types = attr_obj.get("product_types", [])
        product_type_status = {
            ptype: ptype in matched_product_types
            for ptype in original_product_types
        }

        new_obj = attr_obj.copy()
        new_obj["product_types"] = product_type_status

        has_match = any(product_type_status.values())

        if has_match:
            matched[attr_name] = new_obj
        else:
            new_obj["values"] = {v: False for v in attr_obj.get("values", [])}
            not_matched[attr_name] = new_obj

    logging.debug(f"Schema split: {len(matched)} matched attributes, {len(not_matched)} unmatched.")
    return matched, not_matched


def find_key_hits_from_ocr(keys, ocr_results, context_chars=100):
    import re

    word_pattern = re.compile(r"\b\w+\b")

    # Helper: normalize words by lowercasing and stripping trailing non-letters
    def normalize_word(word):
        return re.sub(r'[^a-z]+$', '', word.lower())

    # Build map of normalized key words
    normalized_key_map = {
        tuple(normalize_word(w) for w in word_pattern.findall(key)): key
        for key in keys
    }

    matched_keys = set()
    ocr_key_hit = []

    # Helper: expand to nearest word boundaries
    def expand_to_word_boundaries(text, start, end):
        left = start
        while left > 0 and not re.match(r'\b', text[left-1:left]):
            left -= 1
        right = end
        while right < len(text) and not re.match(r'\b', text[right:right+1]):
            right += 1
        return left, right

    for ocr_idx, result_group in enumerate(ocr_results):
        if not result_group:
            continue

        ocr_result = result_group[0]
        rec_texts = ocr_result.get("rec_texts", [])
        rec_polys = ocr_result.get("rec_polys", [])

        full_text = " ".join(rec_texts)
        text_boundaries = []
        current_pos = 0
        for txt_idx, txt in enumerate(rec_texts):
            text_boundaries.append((
                current_pos,
                current_pos + len(txt),
                txt_idx,
                txt
            ))
            current_pos += len(txt) + 1  # +1 for space

        normalized_full = full_text.lower()
        word_matches = list(word_pattern.finditer(normalized_full))
        if not word_matches:
            continue

        words = [m.group() for m in word_matches]
        spans = [(m.start(), m.end()) for m in word_matches]

        # Normalize OCR words (strip trailing non-letters)
        words_norm = [normalize_word(w) for w in words]

        for key_words, orig_key in normalized_key_map.items():
            key_len = len(key_words)
            if key_len == 0 or key_len > len(words):
                continue

            for i in range(len(words) - key_len + 1):
                candidate_norm = words_norm[i:i + key_len]
                if tuple(candidate_norm) == key_words:
                    start_span = spans[i][0]
                    end_span = spans[i + key_len - 1][1]
                    matched_substring = full_text[start_span:end_span]

                    # Context window
                    raw_start = max(0, start_span - context_chars)
                    raw_end = min(len(full_text), end_span + context_chars)
                    context_start, context_end = expand_to_word_boundaries(full_text, raw_start, raw_end)
                    context_text = full_text[context_start:context_end]

                    # Map back to source line
                    match_text_idx = next(
                        (idx for s, e, idx, _ in text_boundaries if s <= start_span < e),
                        None
                    )
                    bbox = rec_polys[match_text_idx] if match_text_idx is not None and match_text_idx < len(rec_polys) else None

                    matched_keys.add(orig_key)
                    ocr_key_hit.append({
                        "key": orig_key,
                        "matched_substring": matched_substring,
                        "context": context_text,
                        "bbox": bbox,
                        "ocr_result_index": ocr_idx,
                        "text_index": match_text_idx,
                        "char_start": start_span,
                        "char_end": end_span,
                        "context_start": context_start,
                        "context_end": context_end,
                        "source_line": rec_texts[match_text_idx] if match_text_idx is not None else None
                    })

    return matched_keys, ocr_key_hit

# def find_key_hits_from_ocr(keys, ocr_results):
#     logging.debug(f"Searching for {len(keys)} keys in OCR results using exact word matching...")

#     normalized_key_map = {
#         tuple(re.findall(r"\b\w+\b", key.lower())): key
#         for key in keys
#     }

#     matched_keys = set()
#     ocr_key_hit = []

#     for ocr_idx, result_group in enumerate(ocr_results):
#         if not result_group:
#             continue

#         ocr_result = result_group[0]
#         rec_texts = ocr_result.get("rec_texts", [])
#         rec_polys = ocr_result.get("rec_polys", [])

#         for txt_idx, txt in enumerate(rec_texts):
#             words = re.findall(r"\b\w+\b", txt.lower())

#             for key_words, orig_key in normalized_key_map.items():
#                 key_len = len(key_words)

#                 for i in range(len(words) - key_len + 1):
#                     if tuple(words[i:i + key_len]) == key_words:
#                         if orig_key not in matched_keys:
#                             logging.debug(f"✅ Key matched: '{orig_key}' in OCR text: '{txt}'")
#                         matched_keys.add(orig_key)
#                         ocr_key_hit.append({
#                             "key": orig_key,
#                             "text": txt,
#                             "bbox": rec_polys[txt_idx],
#                             "ocr_result_index": ocr_idx,
#                             "text_index": txt_idx
#                         })
#                         break

#     logging.debug(f"Key search complete: {len(matched_keys)} unique key(s) matched.")
#     return matched_keys, ocr_key_hit


def refine_by_key_hits(matched, not_matched, hit_keys):
    logging.info(f"Refining by {len(hit_keys)} OCR-detected keys...")
    key_matched = {}
    key_not_matched = not_matched.copy()

    for attr_name, attr_obj in matched.items():
        if attr_name in hit_keys:
            key_matched[attr_name] = attr_obj
        else:
            new_obj = attr_obj.copy()
            new_obj["values"] = {v: False for v in attr_obj.get("values", [])}
            key_not_matched[attr_name] = new_obj

    logging.debug(f"After key refinement: {len(key_matched)} kept, {len(key_not_matched)} moved to not-matched.")
    return key_matched, key_not_matched

def normalize_space(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def refine_by_value_hits(key_matched, key_not_matched, big_text, extra_values_dict):
    """
    Refines hits by checking standard values and extra suggested values.
    Extra values are only added if they match the big_text.
    """
    value_matched = {}
    value_not_matched = key_not_matched.copy()

    # Normalize big text once
    big_text_norm = normalize_space(big_text)

    if extra_values_dict is None:
        extra_values_dict = {}

    for key in key_matched:
        attr = get_attribute_info_by_key(key, key_matched)

        # --- Standard Values ---
        standard_values = attr.get("values", [])
        value_map = {}
        any_hit = False

        for v in standard_values:
            v_norm = normalize_space(v)
            is_hit = bool(v_norm and v_norm in big_text_norm)
            value_map[v] = is_hit
            if is_hit:
                any_hit = True
                print(f"[VALUE_HIT] {key} -> '{v}': True")

        # --- Extra Values ---
        extra_info = extra_values_dict.get(key, {})
        raw_extras = extra_info.get("possible_extra_values", [])

        # Normalize extras to a list
        possible_extras = []
        if isinstance(raw_extras, list) and len(raw_extras) == 1 and str(raw_extras[0]).startswith('['):
            try:
                possible_extras = json.loads(raw_extras[0])
            except:
                possible_extras = raw_extras
        elif isinstance(raw_extras, str):
            possible_extras = [v.strip() for v in raw_extras.replace(',', '\n').split('\n') if v.strip()]
        else:
            possible_extras = [str(v).strip() for v in raw_extras if str(v).strip()]

        # Clean extra values
        possible_extras = [str(v).strip(' "[]\'') for v in possible_extras]

        # Only add extras if they match
        added_extra_keys = set()
        for extra_val in possible_extras:
            extra_norm = normalize_space(extra_val)
            if extra_norm:
                # Use regex search to be safe
                pattern = re.compile(re.escape(extra_norm))
                if pattern.search(big_text_norm):
                    value_map[extra_val] = True
                    added_extra_keys.add(extra_val)
                    any_hit = True
                    print(f"[EXTRA_HIT] {key} -> '{extra_val}': True")

        # --- Finalize ---
        new_obj = attr.copy()
        new_obj["values"] = value_map
        new_obj["_extra_value_keys"] = added_extra_keys

        # Preserve Expected Output Formatting from original schema object
        expected_output = key_matched[key].get("Expected Output Formatting")
        if expected_output is not None:
            new_obj["Expected Output Formatting"] = expected_output

        if any_hit:
            value_matched[key] = new_obj
            # remove from not_matched if previously there
            value_not_matched.pop(key, None)
        else:
            value_not_matched[key] = new_obj

    return value_matched, value_not_matched



# def refine_by_value_hits(key_matched, key_not_matched, big_text):
#     logging.info("Checking value presence in OCR text...")
#     value_matched = {}
#     # Keep the original not_matched keys as they were
#     value_not_matched = key_not_matched.copy()

#     for key in key_matched:
#         attr = get_attribute_info_by_key(key, key_matched)
#         values = attr.get("values", [])
#         hits = set(find_hits(big_text, values))

#         # Map every value to True or False
#         value_map = {
#             v: str(v).lower() in hits
#             for v in values
#         }

#         new_obj = attr.copy()
#         new_obj["values"] = value_map

#         if any(value_map.values()):
#             value_matched[key] = new_obj
#         else:
#             value_not_matched[key] = new_obj

#     return value_matched, value_not_matched

def search_regex_in_text(regex_string: str, big_text: str):
    if not regex_string:
        return []

    try:
        # Added re.IGNORECASE because your text is lowercase ("input watts")
        pattern = re.compile(regex_string, re.IGNORECASE)
        matches = pattern.findall(big_text)
        return matches
    except re.error as e:
        print(f"Invalid regex: {e}")
        return []

def refine_by_key_value_pair_matching(value_matched, value_not_matched, big_text):
    final_value_matched = {}
    final_value_not_matched = value_not_matched.copy()

    for attr_name, attr_obj in value_matched.items():
        key = attr_obj.get("norm_key") or attr_obj.get("original_key")
        values = attr_obj.get("values", {})
        new_values = {}
        any_value_hit = False

        for value, is_hit in values.items():
            if is_hit and matches_key_value_pair(big_text, key, value):
                new_values[value] = True
                any_value_hit = True
                print(f"✅ Matched key-value pair: {key} -> {value}")
            else:
                new_values[value] = False

        new_attr = attr_obj.copy()
        new_attr["values"] = new_values

        if any_value_hit:
            final_value_matched[attr_name] = new_attr
        else:
            final_value_not_matched[attr_name] = new_attr

    return final_value_matched, final_value_not_matched

# def refine_by_key_value_pair_matching(value_matched, value_not_matched, big_text, regex_withkey):
#     final_value_matched = {}
#     final_value_not_matched = value_not_matched.copy()
#     print(value_matched)
#     for attr_name, attr_obj in value_matched.items():
#         key = attr_obj.get("norm_key") or attr_obj.get("original_key")
#         key_org = attr_obj.get("original_key")
#         values = attr_obj.get("values", {})
#         new_values = {}
#         any_value_hit = False

#         # Step 1: Check existing key-value pair hits
#         for value, is_hit in values.items():
#             if is_hit and matches_key_value_pair(big_text, key, value):
#                 new_values[value] = True
#                 any_value_hit = True
#                 print(f"✅ Matched key-value pair: {key} -> {value}")
#             else:
#                 new_values[value] = False

#         # Step 2: Additional regex-based matching from regex_withkey

#         regex_info = regex_withkey.get(key_org, {})
#         print(key)
#         regex_pattern = regex_info.get("pair_regex")
#         if regex_pattern:
#             # print(f"🔍 Searching for regex pattern: {regex_pattern}")
#             matches_from_regex = search_regex_in_text(regex_pattern, big_text)
#             print(f"🔍 Found matches: {matches_from_regex}")
#             # for match in matches_from_regex:
#             #     # If the matched value exists in values, mark it True
#             #     print(new_values)
#             #     # if match in new_values:
#             #     new_values[match] = True
#             #     any_value_hit = True
#             #     print(f"🔹 Regex matched key-value: {key} -> {match}")
#             for match in matches_from_regex:
#                 # 1. Remove key_org from the match (case-insensitive)
#                 # 2. .strip(" :-") removes leading/trailing colons, dashes, or spaces
#                 cleaned_val = re.sub(re.escape(key_org), "", match, flags=re.IGNORECASE)
#                 cleaned_val = cleaned_val.strip(" :-")

#                 if cleaned_val: # Only add if there's something left after stripping
#                     new_values[cleaned_val] = True
#                     any_value_hit = True
#                     # print(f"🔹 Regex matched & cleaned: {key_org} -> {cleaned_val}")

#         # Save the updated values back to the attribute
#         new_attr = attr_obj.copy()
#         new_attr["values"] = new_values

#         if any_value_hit:
#             final_value_matched[attr_name] = new_attr
#         else:
#             final_value_not_matched[attr_name] = new_attr

#     return final_value_matched, final_value_not_matched

# def refine_by_key_value_pair_matching(value_matched, value_not_matched, big_text):
#     logging.info("Validating key-value co-occurrence in text...")
#     final_value_matched = {}
#     final_value_not_matched = value_not_matched.copy()

#     for attr_name, attr_obj in value_matched.items():
#         key = attr_obj.get("norm_key") or attr_obj.get("original_key")
#         values = attr_obj.get("values", {})
#         new_values = {}
#         any_value_hit = False

#         for value, is_hit in values.items():
#             if is_hit and matches_key_value_pair(big_text, key, value):
#                 new_values[value] = True
#                 any_value_hit = True
#                 logging.info(f"✅ Matched key-value pair: {key} -> {value}")
#             else:
#                 new_values[value] = False

#         new_attr = attr_obj.copy()
#         new_attr["values"] = new_values

#         if any_value_hit:
#             final_value_matched[attr_name] = new_attr
#         else:
#             final_value_not_matched[attr_name] = new_attr

#     logging.info(f"Key-value validation complete: {len(final_value_matched)} attributes confirmed.")
#     return final_value_matched, final_value_not_matched
