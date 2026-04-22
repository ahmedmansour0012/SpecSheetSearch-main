import logging
import os
import json
import fitz
from PIL import Image
import json
import logging

def save_ocr_debug(ocr_results, big_text, output_dir, base_name):
    """
    Save raw OCR results (per page) and combined text to debug files.
    Polys are serialized as nested lists so the output is JSON-safe.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Raw per-page OCR ---
    pages_out = []
    for page_idx, result_group in enumerate(ocr_results):
        if not result_group or "rec_texts" not in result_group[0]:
            pages_out.append({"page": page_idx, "rec_texts": [], "rec_scores": [], "rec_polys": []})
            continue
        page = result_group[0]
        polys = page.get("rec_polys", [])
        serialized_polys = [
            [[float(coord) for coord in point] for point in poly]
            for poly in polys
        ]
        pages_out.append({
            "page": page_idx,
            "rec_texts": page.get("rec_texts", []),
            "rec_scores": [float(s) for s in page.get("rec_scores", [])],
            "rec_polys": serialized_polys,
        })

    raw_path = os.path.join(output_dir, f"ocr_raw_{base_name}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(pages_out, f, indent=2, ensure_ascii=False)
    logging.info(f"  → OCR raw saved to: {raw_path}")

    # --- Combined text ---
    combined_path = os.path.join(output_dir, f"ocr_combined_{base_name}.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(big_text)
    logging.info(f"  → OCR combined text saved to: {combined_path}")


def convert_pdf_with_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pixmap = page.get_pixmap(dpi=300)

        # Convert pixmap to a PIL Image object
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        images.append(img)

    doc.close()
    return images

def load_attribute_schema(file_path):
    """Load the attribute schema from a pure JSON file."""
    logging.info(f"Loading attribute schema from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        logging.info(f"Successfully loaded schema with {len(schema)} attributes.")
        return schema
    except FileNotFoundError:
        logging.error(f"Schema file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in schema file {file_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading schema from {file_path}: {e}")
        raise

def get_attribute_info_by_key(search_key, schema):
    """
    Look up an attribute by normalized key (case-insensitive, stripped).
    Returns dict with 'data_type', 'values' (normalized), and 'product_types',
    or None if not found.
    """
    search_key_norm = search_key.strip().lower()
    logging.debug(f"Searching for attribute key: '{search_key}' (normalized: '{search_key_norm}')")

    for orig_key, attr in schema.items():
        if orig_key.strip().lower() == search_key_norm:
            logging.debug(f"Match found for key: '{search_key}' → original key: '{orig_key}'")
            result = {
                "original_key": orig_key,
                "norm_key": orig_key.strip().lower(),
                "data_type": attr["data_type"],
                "values": [v.strip().lower() for v in attr["values"]],
                "product_types": attr["product_types"]
            }
            return result

    logging.warning(f"No matching attribute found for key: '{search_key}'")
    return None

def save_llm_output(response, output_dir="final_result", base_name="output"):
    """
    Saves a Python dict as a JSON file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{base_name}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=4, ensure_ascii=False)

    logging.info(f"LLM output saved to: {output_path}")
    return response


def save_final_result(final_result, output_dir="final_result", base_name="output"):
    output_path = os.path.join(output_dir, f"final_result_{base_name}.json")

    os.makedirs(output_dir, exist_ok=True)

    # Strip false values from the values dict before saving
    cleaned = {}
    for attr_name, attr_obj in final_result.items():
        new_attr = {k: v for k, v in attr_obj.items() if k not in ("values", "_extra_value_keys")}
        raw_values = attr_obj.get("values", {})
        extra_keys = attr_obj.get("_extra_value_keys", set())
        if isinstance(raw_values, dict):
            new_attr["values"] = {
                v: state for v, state in raw_values.items()
                if state is True or v not in extra_keys
            }
        else:
            new_attr["values"] = raw_values
        cleaned[attr_name] = new_attr

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    logging.info(f"Final result saved to: {output_path}")
    
def merge_match_results(*result_pairs):
    """
    Merge multiple (matched, not_matched) result pairs.

    Rules:
    - If an attribute is matched by ANY strategy → final_matched
    - Attribute can never appear in both outputs
    """

    final_matched = {}
    final_not_matched = {}

    all_keys = set()

    # Collect all keys seen anywhere
    for matched, not_matched in result_pairs:
        all_keys.update(matched.keys())
        all_keys.update(not_matched.keys())

    for key in all_keys:
        matched_versions = [
            matched[key]
            for matched, _ in result_pairs
            if key in matched
        ]

        if matched_versions:
            # Take the first matched version (they should be equivalent)
            final_matched[key] = matched_versions[0]
        else:
            # Guaranteed not matched in all strategies
            for _, not_matched in result_pairs:
                if key in not_matched:
                    final_not_matched[key] = not_matched[key]
                    break

    return final_matched, final_not_matched
  