from paddleocr import PaddleOCR
import re
import numpy as np

def get_ocr_object_per_page(images,ocr):
  ocr_results =[]
  for image in images:
    np_img = np.asarray(image)
    res = ocr.predict(np_img)
    if res:
      ocr_results.append(res)
  return ocr_results

def build_full_ocr_text(ocr_results):
    # Collect all text snippets into a list first
    all_snippets = []

    for result in ocr_results:
        # Use .get() or check if result is valid to prevent crashes
        if result and 'rec_texts' in result[0]:
            snippet = " ".join(result[0]['rec_texts']).lower()
            all_snippets.append(snippet)

    # Join everything with a space (or "\n" for new lines) at the end
    return " ".join(all_snippets)
# def build_full_ocr_text(ocr_results):
#     big_text = ''
#     for i in range(len(ocr_results)):
#         rec_texts = ocr_results[i][0]['rec_texts']
#         big_text += " ".join(rec_texts).lower()
#     logging.debug(f"Built full OCR text (length: {len(big_text)} characters)")
#     return big_text

def filter_ocr_key_hit_by_value_matched(ocr_key_hit, value_matched):
    """
    Return ALL OCR hits whose keys appear in value_matched.
    """
    # Build a set of keys we care about (from value_matched's original_key)
    keys_of_interest = {
        vm_data["original_key"] for vm_data in value_matched.values()
    }

    # Collect all OCR hits that match any of these keys
    filtered = [
        hit for hit in ocr_key_hit
        if hit.get("key") in keys_of_interest
    ]

    return filtered

def filter_ocr_keys_by_regions(ocr_key_hits, regions_by_page):
    """
    Filter OCR key hits by a list of region dicts grouped by page index.
    Adds 'stop_y' to each hit, which is the bottom Y coordinate of the overlapping region.

    Args:
        ocr_key_hits: list of dicts, each with 'ocr_result_index' and 'bbox'
        regions_by_page: dict mapping page_index -> list of regions
            Each region is {'x1', 'y1', 'x2', 'y2'}

    Returns:
        list of OCR key hits that overlap at least one region on their page,
        with an extra field 'stop_y' = region['y2']
    """

    def poly_to_bbox(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    def boxes_intersect(a, b):
        return not (
            a[2] < b[0] or
            a[0] > b[2] or
            a[3] < b[1] or
            a[1] > b[3]
        )

    filtered = []

    for hit in ocr_key_hits:
        page_idx = hit.get("ocr_result_index")
        poly = hit.get("bbox")
        if poly is None or page_idx not in regions_by_page:
            continue

        key_box = poly_to_bbox(poly)

        # check if overlaps any region on this page
        for region in regions_by_page[page_idx]:
            region_box = (region['x1'], region['y1'], region['x2'], region['y2'])
            if boxes_intersect(key_box, region_box):
                # add stop_y to the hit
                hit_copy = hit.copy()
                hit_copy['stop_y'] = region['y2']
                filtered.append(hit_copy)
                break  # one match is enough

    return filtered

def normalize_space(text):
    return re.sub(r'\s+', ' ', text).strip().lower()
 
def match_values_for_keys(
    row_for_key_data,
    value_matched,
    value_not_matched,
    extra_values_dict=None
):
    """
    Match OCR-extracted text under each key against allowed values.
    Extra values are only added if they actually match OCR text.
    """
    if value_not_matched is None:
        value_not_matched = {}

    if extra_values_dict is None:
        extra_values_dict = {}

    # Build lookup: key -> combined OCR text
    ocr_text_by_key = {}
    for row in row_for_key_data:
        key = row["key"]
        row_text = " ".join(t["text"] for t in row.get("text", [])).lower()
        ocr_text_by_key[key] = ocr_text_by_key.get(key, "") + (" " + row_text if key in ocr_text_by_key else row_text)

    final_value_matched = {}
    final_value_not_matched = value_not_matched.copy()

    for attr_name, attr_obj in value_matched.items():
        key = attr_obj.get("original_key") or attr_name
        standard_values = attr_obj.get("values", {})
        combined_text = ocr_text_by_key.get(key, "")

        # --- Handle Extra Values ---
        extra_info = extra_values_dict.get(attr_name, {})
        raw_extras = extra_info.get("possible_extra_values", [])

        # Normalize extras into a clean list
        possible_extras = []
        if isinstance(raw_extras, str):
            possible_extras = [v.strip() for v in raw_extras.replace(',', '\n').split('\n') if v.strip()]
        elif isinstance(raw_extras, list):
            if len(raw_extras) == 1 and isinstance(raw_extras[0], str) and raw_extras[0].startswith('['):
                try:
                    possible_extras = json.loads(raw_extras[0])
                except:
                    possible_extras = raw_extras
            else:
                possible_extras = [str(v).strip() for v in raw_extras if str(v).strip()]

        # Clean extra values
        possible_extras = [str(v).strip(' "[]\'') for v in possible_extras]

        new_values = {}
        any_value_hit = False
        combined_norm = normalize_space(combined_text)

        # 1️⃣ Process Standard Values (always include True/False)
        for val in standard_values.keys():
            val_norm = normalize_space(val)
            is_hit = bool(val_norm and val_norm in combined_norm)
            new_values[val] = is_hit
            if is_hit:
                any_value_hit = True

        # 2️⃣ Process Extra Values (only add if matched)
        for extra_val in possible_extras:
            extra_norm = normalize_space(extra_val)
            if extra_norm and extra_norm in combined_norm:
                new_values[extra_val] = True
                any_value_hit = True
                # no False entries for extras

        # 3️⃣ Add to final dicts
        new_attr = attr_obj.copy()
        new_attr["values"] = new_values

        if any_value_hit:
            final_value_matched[attr_name] = new_attr
        else:
            final_value_not_matched[attr_name] = new_attr

    return final_value_matched, final_value_not_matched
