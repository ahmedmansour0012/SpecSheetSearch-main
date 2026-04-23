from paddleocr import PaddleOCR
import re
import logging
import numpy as np
# import os
# from pathlib import Path
# from mistralai.client import Mistral

# Phrases that unambiguously mark a page as photometric test data
_PHOTOMETRIC_MARKERS = [
    "candlepower",
    "zonal lumens",
    "coefficient of utilization",
    "footcandle plot",
    "spacing criterion",
    "angle in degrees",
    "mounting height aff",
    "ies file",
    "bug rating",
    "candela distribution",
    "photometric report",
    "luminaire efficiency",
    "fc per w",
    "luminance",
    "nadir",
    "photometric data",
]

# Phrases that anchor a page as a product spec / ordering page
_SPEC_MARKERS = [
    "ordering information",
    "ordering code",
    "catalog number",
    "specifications",
    "features",
    "options",
    "accessories",
    "warranty",
    "finish",
]

def classify_page_content(page_ocr_result) -> str:
    """
    Classify a single page OCR result as 'spec', 'photometric', or 'mixed'.

    Returns 'photometric' only when strong photometric evidence exists and no
    spec markers are present, so we never accidentally discard ordering pages.
    """
    if not page_ocr_result or "rec_texts" not in page_ocr_result[0]:
        return "spec"

    page_text = " ".join(page_ocr_result[0]["rec_texts"]).lower()

    photo_hits = sum(1 for m in _PHOTOMETRIC_MARKERS if m in page_text)
    spec_hits  = sum(1 for m in _SPEC_MARKERS if m in page_text)

    if photo_hits >= 2 and spec_hits == 0:
        return "photometric"
    if photo_hits >= 1 and spec_hits == 0:
        return "mixed"
    return "spec"


def filter_spec_pages(images, ocr_results):
    """
    Drop pages classified as purely photometric.
    Always keeps at least the first page and never returns an empty list.
    """
    pairs = list(zip(images, ocr_results))
    kept = [(img, res) for img, res in pairs if classify_page_content(res) != "photometric"]

    if not kept:
        logging.warning("classify_page_content dropped ALL pages — keeping first page as fallback.")
        kept = [pairs[0]]

    dropped = len(pairs) - len(kept)
    if dropped:
        logging.info(f"  → Page classifier dropped {dropped} photometric page(s) from pipeline.")

    filtered_images, filtered_ocr = zip(*kept)
    return list(filtered_images), list(filtered_ocr)


def get_ocr_object_per_page_paddle(images, ocr_engine):
    ocr_results = []
    for image in images:
        np_img = np.asarray(image)
        res = ocr_engine.predict(np_img)
        if res:
            ocr_results.append(res)
    return ocr_results


# def get_ocr_object_per_page_mistral(pdf_path, images, api_key=None):
#     """
#     Mistral OCR backend. Uploads the PDF once, returns the same structure as
#     get_ocr_object_per_page_paddle:
#       [ [{"rec_texts": [...], "rec_polys": [...], "rec_scores": [...]}], ... ]
#
#     images provides per-page pixel dimensions for synthetic rec_polys so that
#     downstream spatial functions stay in the same coordinate space as YOLO.
#     """
#     if api_key is None:
#         api_key = os.environ.get("MISTRAL_API_KEY")
#
#     client = Mistral(api_key=api_key)
#
#     pdf_bytes = Path(pdf_path).read_bytes()
#     uploaded = client.files.upload(
#         file={"file_name": Path(pdf_path).name, "content": pdf_bytes},
#         purpose="ocr",
#     )
#     signed_url = client.files.get_signed_url(file_id=uploaded.id, expiry=1)
#     response = client.ocr.process(
#         model="mistral-ocr-latest",
#         document={"type": "document_url", "document_url": signed_url.url},
#     )
#     response_dict = response.model_dump()
#     pages = response_dict.get("pages", [])
#
#     ocr_results = []
#     for page_idx, page in enumerate(pages):
#         w, h = images[page_idx].size if page_idx < len(images) else (1000, 1000)
#
#         markdown = page.get("markdown", "")
#         lines = [ln.strip() for ln in markdown.split("\n") if ln.strip()]
#
#         n = len(lines)
#         if n:
#             rec_polys = [
#                 [[0, int(li * h / n)], [w, int(li * h / n)],
#                  [w, int((li + 1) * h / n)], [0, int((li + 1) * h / n)]]
#                 for li in range(n)
#             ]
#             rec_scores = [1.0] * n
#         else:
#             rec_polys, rec_scores = [], []
#
#         ocr_results.append([{
#             "rec_texts": lines,
#             "rec_polys": rec_polys,
#             "rec_scores": rec_scores,
#         }])
#
#     return ocr_results

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
                hit_copy = hit.copy()
                hit_copy['stop_y'] = region['y2']
                hit_copy['table_y1'] = region['y1']
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
                # print(f"[TABLE_HIT] {attr_name} -> '{val}': True")

        # 2️⃣ Process Extra Values (only add if matched)
        for extra_val in possible_extras:
            extra_norm = normalize_space(extra_val)
            if extra_norm and extra_norm in combined_norm:
                new_values[extra_val] = True
                any_value_hit = True
                print(f"[TABLE_EXTRA_HIT] {attr_name} -> '{extra_val}': True")
                # no False entries for extras

        # 3️⃣ Add to final dicts
        new_attr = attr_obj.copy()
        new_attr["values"] = new_values

        if any_value_hit:
            final_value_matched[attr_name] = new_attr
        else:
            final_value_not_matched[attr_name] = new_attr

    return final_value_matched, final_value_not_matched
