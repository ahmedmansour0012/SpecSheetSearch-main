import os
import json
import logging
import re
from html.parser import HTMLParser

import numpy as np

from model_loader import get_table_structure_instance
from ocr import normalize_space

logger = logging.getLogger(__name__)


class _SLANetTableParser(HTMLParser):
    """Walk the SLANet HTML token stream and emit (row, col, rowspan, colspan) per <td>."""

    def __init__(self):
        super().__init__()
        self.cells = []
        self.row = -1
        self.col_cursor = 0
        self._occupied = set()

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row += 1
            self.col_cursor = 0
            return
        if tag != "td":
            return

        attr = dict(attrs)
        rowspan = int(attr.get("rowspan", 1) or 1)
        colspan = int(attr.get("colspan", 1) or 1)

        while (self.row, self.col_cursor) in self._occupied:
            self.col_cursor += 1

        self.cells.append({
            "row": self.row,
            "col": self.col_cursor,
            "rowspan": rowspan,
            "colspan": colspan,
        })

        for r in range(self.row, self.row + rowspan):
            for c in range(self.col_cursor, self.col_cursor + colspan):
                self._occupied.add((r, c))

        self.col_cursor += colspan


def _parse_structure_tokens(structure_tokens):
    """Join SLANet structure tokens into HTML and parse into ordered cell metadata."""
    html = "".join(t for t in structure_tokens if isinstance(t, str))
    parser = _SLANetTableParser()
    parser.feed(html)
    return parser.cells, html


def _poly_center(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, min(ys), min(xs)


def _fill_cell_text(cells, rec_texts, rec_polys):
    """
    Assign OCR text to each cell by center-point containment: an OCR poly is
    attributed to a cell when its center falls inside the cell's page-coord bbox.
    Within a cell, pieces are concatenated top-to-bottom, left-to-right.
    """
    for cell in cells:
        bbox = cell.get("bbox_page")
        if not bbox:
            cell["text"] = ""
            continue
        cx1, cy1, cx2, cy2 = bbox
        hits = []
        for text, poly in zip(rec_texts, rec_polys):
            center_x, center_y, top_y, left_x = _poly_center(poly)
            if cx1 <= center_x <= cx2 and cy1 <= center_y <= cy2:
                hits.append((top_y, left_x, text))
        hits.sort()
        cell["text"] = " ".join(h[2] for h in hits)


def _run_slanet_on_crop(model, crop_np):
    """Run SLANet_plus on a cropped table image, returning (tokens, cell_bboxes, score)."""
    outputs = list(model.predict(crop_np, batch_size=1))
    if not outputs:
        return None
    res = outputs[0]

    # SLANet result exposes fields via attribute or dict-like __getitem__ depending on version.
    def _get(field):
        if hasattr(res, field):
            return getattr(res, field)
        try:
            return res[field]
        except Exception:
            return None

    tokens = _get("structure")
    bboxes = _get("bbox")
    score = _get("structure_score")

    if bboxes is None or tokens is None:
        return None

    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.tolist()

    return tokens, bboxes, float(score) if score is not None else None


def extract_structured_tables(images, regions_by_page, ocr_results):
    """
    For every YOLO-detected table region, run SLANet_plus, translate cell bboxes
    into page coordinates, and fill each cell with the OCR text whose center
    falls inside it.

    Returns a list of page dicts: [{"page_index": int, "tables": [...]}, ...]
    """
    if not regions_by_page:
        return []

    model = get_table_structure_instance()
    pages_out = []

    for page_idx in sorted(regions_by_page.keys()):
        page_image = images[page_idx]
        page_ocr = ocr_results[page_idx][0] if ocr_results[page_idx] else {}
        rec_texts = page_ocr.get("rec_texts", [])
        rec_polys = page_ocr.get("rec_polys", [])

        tables_out = []
        for region in regions_by_page[page_idx]:
            x1, y1 = int(region["x1"]), int(region["y1"])
            x2, y2 = int(region["x2"]), int(region["y2"])
            if x2 <= x1 or y2 <= y1:
                continue

            crop_pil = page_image.crop((x1, y1, x2, y2))
            crop_np = np.asarray(crop_pil)

            try:
                result = _run_slanet_on_crop(model, crop_np)
            except Exception as exc:
                logger.warning(f"SLANet_plus failed on page {page_idx} table {[x1,y1,x2,y2]}: {exc}")
                continue

            if result is None:
                continue

            tokens, cell_bboxes_crop, score = result
            cells, html = _parse_structure_tokens(tokens)

            for cell, crop_box in zip(cells, cell_bboxes_crop):
                if crop_box is None or len(crop_box) < 4:
                    cell["bbox_page"] = None
                    continue
                bx1, by1, bx2, by2 = crop_box[:4]
                cell["bbox_page"] = [
                    float(bx1 + x1),
                    float(by1 + y1),
                    float(bx2 + x1),
                    float(by2 + y1),
                ]

            _fill_cell_text(cells, rec_texts, rec_polys)

            for cell in cells:
                cell["is_header"] = cell["row"] == 0

            low_confidence = score is not None and score < 0.5
            tables_out.append({
                "table_bbox_page": [float(x1), float(y1), float(x2), float(y2)],
                "html": html,
                "structure_score": score,
                "low_confidence": low_confidence,
                "cells": cells,
            })

        if tables_out:
            pages_out.append({"page_index": page_idx, "tables": tables_out})

    return pages_out


def save_structured_tables_debug(structured_tables, output_dir, base_name):
    """Write the structured-table result to <output_dir>/tables_raw_<base_name>.json."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"tables_raw_{base_name}.json")
    payload = {"pdf": f"{base_name}.pdf", "pages": structured_tables}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"  → Structured tables saved to: {path}")
    return path


def _iter_all_cells(structured_tables):
    for page in structured_tables:
        for table in page["tables"]:
            if table.get("low_confidence"):
                continue
            for cell in table["cells"]:
                yield page["page_index"], table, cell


def _cells_in_same_column_below(table, header_cell):
    header_col_range = range(header_cell["col"], header_cell["col"] + header_cell["colspan"])
    header_bottom_row = header_cell["row"] + header_cell["rowspan"]
    below = []
    for cell in table["cells"]:
        if cell["row"] < header_bottom_row:
            continue
        cell_col_range = range(cell["col"], cell["col"] + cell["colspan"])
        if set(header_col_range).intersection(cell_col_range):
            below.append(cell)
    return below


def _cells_in_same_row_right(table, label_cell):
    label_row_range = range(label_cell["row"], label_cell["row"] + label_cell["rowspan"])
    label_right_col = label_cell["col"] + label_cell["colspan"]
    right = []
    for cell in table["cells"]:
        if cell["col"] < label_right_col:
            continue
        cell_row_range = range(cell["row"], cell["row"] + cell["rowspan"])
        if set(label_row_range).intersection(cell_row_range):
            right.append(cell)
    return right


def _collect_candidate_text_for_key(structured_tables, key):
    """
    Find every cell whose text contains the key. For each such cell, treat it
    as a header (if row 0) or a row label (otherwise) and gather candidate
    value cells accordingly. Returns a single concatenated lowercased string
    of all candidate values.
    """
    key_norm = normalize_space(key)
    if not key_norm:
        return ""

    chunks = []
    for _page_idx, table, cell in _iter_all_cells(structured_tables):
        cell_text_norm = normalize_space(cell.get("text", ""))
        if not cell_text_norm or key_norm not in cell_text_norm:
            continue

        if cell["is_header"] or cell["row"] == 0:
            value_cells = _cells_in_same_column_below(table, cell)
        else:
            value_cells = _cells_in_same_row_right(table, cell)

        for vc in value_cells:
            vt = vc.get("text", "").strip()
            if vt:
                chunks.append(vt)

    return " ".join(chunks).lower()


def match_keys_against_structured_tables(
    structured_tables,
    value_matched,
    value_not_matched,
    extra_values_dict=None,
):
    """
    Structured-table matching strategy. Mirrors match_values_for_keys in ocr.py
    but draws candidate text from cell-level structure instead of vertical OCR
    strips. Returns (matched, not_matched) in the same shape as the other
    strategies so merge_match_results can consume it directly.
    """
    if value_not_matched is None:
        value_not_matched = {}
    if extra_values_dict is None:
        extra_values_dict = {}

    final_matched = {}
    final_not_matched = dict(value_not_matched)

    if not structured_tables:
        return final_matched, dict(value_matched) | final_not_matched

    for attr_name, attr_obj in value_matched.items():
        key = attr_obj.get("original_key") or attr_name
        standard_values = attr_obj.get("values", {})

        combined_text = _collect_candidate_text_for_key(structured_tables, key)

        raw_extras = extra_values_dict.get(attr_name, {}).get("possible_extra_values", [])
        possible_extras = []
        if isinstance(raw_extras, str):
            possible_extras = [v.strip() for v in raw_extras.replace(",", "\n").split("\n") if v.strip()]
        elif isinstance(raw_extras, list):
            possible_extras = [str(v).strip(' "[]\'') for v in raw_extras if str(v).strip()]

        new_values = {}
        any_hit = False
        combined_norm = normalize_space(combined_text)

        for val in standard_values.keys():
            val_norm = normalize_space(val)
            is_hit = bool(val_norm and val_norm in combined_norm)
            new_values[val] = is_hit
            if is_hit:
                any_hit = True

        for extra_val in possible_extras:
            extra_norm = normalize_space(extra_val)
            if extra_norm and extra_norm in combined_norm:
                new_values[extra_val] = True
                any_hit = True

        new_attr = attr_obj.copy()
        new_attr["values"] = new_values

        if any_hit:
            final_matched[attr_name] = new_attr
        else:
            final_not_matched[attr_name] = new_attr

    return final_matched, final_not_matched
