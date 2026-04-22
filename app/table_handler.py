import logging
from doclayout_yolo import YOLOv10
from  model_loader import get_yolo_model_path
from ocr import filter_ocr_keys_by_regions, match_values_for_keys

# def layout_detect(model,images):
#     det_res = model.predict(
#       images,   # Image to predict
#       imgsz=1024,        # Prediction image size
#       conf=0.1,          # Confidence threshold
#       device="cpu" ,   # Device to use (e.g., 'cuda:0' or 'cpu')
#       )
#     return det_res

import logging
logger = logging.getLogger(__name__)

def layout_detect(images):
    """
    Detect layout elements in one or more PIL images.

    Args:
        images: Single PIL image or list of PIL images

    Returns:
        List of detection dictionaries per page
    """
    logger.info("Starting layout detection")

    try:
        filepath = get_yolo_model_path()
        model = YOLOv10(filepath)
        logger.debug(f"Model loaded successfully. Running detection on {len(images)} image(s)")

        det_res = model.predict(
            images,   # Image to predict
            imgsz=1024,        # Prediction image size
            conf=0.05,                           # Confidence threshold
            device='cpu' ,    # Device to use (e.g., 'cuda:0' or 'cpu')
        )

        logger.info(f"Layout detection completed for {len(det_res)} page(s)")
        return det_res

    except Exception as e:
        logger.error(f"Layout detection failed: {str(e)}", exc_info=True)
        raise

def get_text_under_key_to_page_end(
    key_poly,
    rec_texts,
    rec_polys,
    image_height=None,
    stop_y=None,
    horizontal_tolerance=0.3
):
    """
    Extract all OCR text under a key.

    Args:
        key_poly: polygon of the key
        rec_texts: all OCR texts on page
        rec_polys: all OCR polys on page
        image_height: optional image height (preferred)
        stop_y: optional Y coordinate to stop extraction
        horizontal_tolerance: required horizontal overlap ratio
    """

    def poly_to_bbox(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    key_x1, key_y1, key_x2, key_y2 = poly_to_bbox(key_poly)
    key_width = key_x2 - key_x1

    # Determine bottom boundary
    if stop_y is not None:
        page_bottom = stop_y
    elif image_height is not None:
        page_bottom = image_height
    else:
        page_bottom = max(poly_to_bbox(p)[3] for p in rec_polys)

    hits = []

    for txt, poly in zip(rec_texts, rec_polys):
        x1, y1, x2, y2 = poly_to_bbox(poly)

        # must be below the key
        if y1 <= key_y2:
            continue

        # must be above stop boundary
        if y1 >= page_bottom:
            continue

        # horizontal overlap
        overlap_x1 = max(key_x1, x1)
        overlap_x2 = min(key_x2, x2)
        overlap_width = max(0, overlap_x2 - overlap_x1)

        if overlap_width / key_width >= horizontal_tolerance:
            hits.append({
                "text": txt,
                "bbox": poly,
                "y_top": y1
            })

    hits.sort(key=lambda x: x["y_top"])
    return hits

def detect_table_regions_for_key_hits(filtered_keys,  images):
    """
    Given a list of OCR key hits and value-matched attributes, detect layout tables
    on the relevant pages and return table bounding boxes grouped by relative page index.

    Args:
        ocr_key_hit (list): List of key-hit dicts from find_key_hits_from_ocr.
        value_matched (dict): Attributes that passed value-presence check.
        layout_model: Layout detection model (e.g., YOLO).
        images (list): Full list of PDF page images.

    Returns:
        dict: {relative_page_index (int): [table_bbox1, table_bbox2, ...]}
    """
    # Find which pages contain these hits
    if not filtered_keys:
        return {}

    # Find which pages contain these hits
    page_indices = {item["ocr_result_index"] for item in filtered_keys}
    min_page = min(page_indices)
    max_page = max(page_indices)

    logging.info(f"  → Detecting layout on pages {min_page + 1} to {max_page + 1} (0-indexed: {min_page}-{max_page})...")

    # Run layout detection only on relevant pages
    image_subset = images[min_page : max_page + 1]
    layout_results = layout_detect(image_subset)

    regions_by_page = {}
    for i, res in enumerate(layout_results):
        d = res.summary()
        d_sorted = sorted(d, key=lambda x: (x['box']['y1'], x['box']['x1']))

        page_tables = []
        for det in d_sorted:
            if det['class'] == 5:  # "Table" class
                bbox = det["box"]
                page_tables.append(bbox)

        if page_tables:
            # Map back to original page index
            original_page_idx = min_page + i
            regions_by_page[original_page_idx] = page_tables

    logging.debug(f"Detected tables on {len(regions_by_page)} page(s).")
    return regions_by_page

def extract_candidate_rows_for_keys(filtered_keys, ocr_results):
    """
    For each filtered key hit, extract the text that appears under the key
    (from the key's bounding box down to a stopping Y-coordinate) on its page.

    Args:
        filtered_keys (list): List of filtered key-hit dicts from `filter_ocr_keys_by_regions`.
            Each must contain: 'key', 'bbox', 'ocr_result_index', and 'stop_y'.
        ocr_results (list): Full OCR output per page (from `get_ocr_object_per_page`).

    Returns:
        tuple:
            - pages (set of int): Page indices involved.
            - row_for_key_data (list of dict): Each with 'key', 'page', and 'text' (list of strings).
    """
    pages = set()
    row_for_key_data = []

    logging.debug(f"Extracting candidate text rows for {len(filtered_keys)} filtered keys...")

    for item in filtered_keys:
        index = item["ocr_result_index"]
        key_poly = item["bbox"]
        table_y1 = item.get("table_y1")
        table_y2 = item.get("stop_y")

        # Header position check: key must sit in the top 25% of the table.
        # If table bounds are known and the key is deeper in the body, it is
        # a data cell (e.g. photometric row label) not a column header — skip it.
        if table_y1 is not None and table_y2 is not None:
            table_height = table_y2 - table_y1
            if table_height > 0:
                key_y1 = min(p[1] for p in key_poly)
                header_ratio = (key_y1 - table_y1) / table_height
                if header_ratio > 0.25:
                    logging.debug(
                        f"Skipping key '{item['key']}' — not a table header "
                        f"(position {header_ratio:.0%} from table top)."
                    )
                    continue

        pages.add(index)
        page_ocr = ocr_results[index][0]
        results = get_text_under_key_to_page_end(
            key_poly=key_poly,
            rec_texts=page_ocr["rec_texts"],
            rec_polys=page_ocr["rec_polys"],
            stop_y=table_y2
        )
        row_for_key_data.append({
            'key': item['key'],
            'page': index,
            'text': results,
        })

    logging.debug(f"Extracted candidate rows from {len(pages)} page(s).")
    return pages, row_for_key_data


def filter_ocr_keys_and_match_values(
    ocr_key_hit,
    value_matched,
    value_not_matched,
    regions_by_page,
    row_for_key_data
):
    """
    Filters OCR key hits by table regions and performs final value matching.

    Args:
        ocr_key_hit (list): Full list of detected key hits.
        value_matched (dict): Attributes that passed value-presence test.
        value_not_matched (dict): Attributes that failed value-presence.
        regions_by_page (dict): Table regions per page (from layout detection).
        row_for_key_data: Data structure mapping keys to candidate table rows.

    Returns:
        tuple: (final_value_matched, final_value_not_matched)
    """
    # Step 1: Filter key hits by table regions
    filtered_keys = filter_ocr_keys_by_regions(ocr_key_hit, regions_by_page)
    logging.debug(f"Filtered to {len(filtered_keys)} key hits inside table regions.")

    # Step 2: Final value matching using table row context
    final_value_matched, final_value_not_matched = match_values_for_keys(
        row_for_key_data,
        value_matched,
        value_not_matched
    )
    return final_value_matched, final_value_not_matched