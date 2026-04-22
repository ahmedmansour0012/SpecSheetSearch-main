import os
import logging
import json
from  model_loader import get_qwen_model_path, get_yolo_model_path
from  input_handler import convert_pdf_with_pymupdf
from ocr import get_ocr_object_per_page_paddle, get_ocr_object_per_page_mistral
from  generate_mouting import (
    build_mounting_prompt,
    get_valid_json,
    load_schema_and_derive_product_types,)
from  input_handler import save_final_result, merge_match_results, save_ocr_debug
from  ocr import build_full_ocr_text, filter_ocr_key_hit_by_value_matched, filter_ocr_keys_by_regions, match_values_for_keys, filter_spec_pages
from  serching import (
    match_product_types_via_lookup,
    split_schema_by_product_type_match, 
    find_key_hits_from_ocr,
    refine_by_key_hits,
    refine_by_value_hits,
    refine_by_key_value_pair_matching)
from  table_handler import (
    detect_table_regions_for_key_hits,
    extract_candidate_rows_for_keys,)
from  generate_mouting import generate_llm_response
from ocr import build_full_ocr_text
from extract_txt_form_expected import format_row_data_to_markdown, get_value_prompt, build_ocr_context

def process_lighting_spec_sheet(pdf_path, schema_path, output_dir="final_result", use_gpu=False, ocr_backend="paddle", ocr_engine=None):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    logging.info(f"📄 Processing spec sheet: {base_name}.pdf")

    # Step 1: PDF to images
    logging.info("  → Converting PDF to images...")
    images = convert_pdf_with_pymupdf(pdf_path)
    images = images[:3]   # Limit to first 3 pages for efficiency; adjust as needed

    # Step 2: OCR
    logging.info(f"  → Running OCR on all pages (backend: {ocr_backend})...")
    if ocr_backend == "mistral":
        ocr_results = get_ocr_object_per_page_mistral(pdf_path, images)
        ocr_results = ocr_results[:len(images)]  # align page count
    else:
        ocr_results = get_ocr_object_per_page_paddle(images, ocr_engine)

    # Step 2.5: Drop photometric-only pages (contamination guard)
    logging.info("  → Classifying pages and filtering photometric data pages...")
    images, ocr_results = filter_spec_pages(images, ocr_results)

    # Step 3: Load schema & derive product types
    logging.info("  → Loading attribute schema and deriving product types...")
    schema, product_type_set = load_schema_and_derive_product_types(schema_path)

    llm_output_path = os.path.join(output_dir, "llm_output.json")
    lookup = {}

    if os.path.exists(llm_output_path):
        logging.info(f"  → Found cached LLM output: {llm_output_path}")

        with open(llm_output_path, "r", encoding="utf-8") as f:
            lookup = json.load(f)

        existing_product_types = set(lookup.keys())

        # Case 1: Perfect match → reuse
        if existing_product_types == product_type_set:
            logging.info("  → Cached lookup matches schema product types. Reusing.")
        else:
            # Case 2: Partial match → generate missing only
            missing_product_types = product_type_set - existing_product_types

            if missing_product_types:
                logging.info(
                    f"  → Generating lookup for missing product types: "
                    f"{sorted(missing_product_types)}"
                )

                prompt = build_mounting_prompt(missing_product_types)

                new_entries = get_valid_json(
                    prompt=prompt,
                    use_gpu=use_gpu
                )

                if not isinstance(new_entries, dict):
                    raise ValueError("LLM output must be a dict of product_type -> mounting")

                # Append
                lookup.update(new_entries)

                # Save updated lookup
                with open(llm_output_path, "w", encoding="utf-8") as f:
                    json.dump(lookup, f, indent=2)

                logging.info("  → Lookup updated and saved.")
    else:
        # Case 3: No cache → generate everything
        logging.info("  → No cached lookup found. Generating full lookup...")

        prompt = build_mounting_prompt(product_type_set)

        lookup = get_valid_json(
            prompt=prompt,
            use_gpu=use_gpu
        )

        if not isinstance(lookup, dict):
            raise ValueError("LLM output must be a dict of product_type -> mounting")

        with open(llm_output_path, "w", encoding="utf-8") as f:
            json.dump(lookup, f, indent=2)

        logging.info("  → Full lookup generated and saved.")


    # Step 5: Build full OCR text
    logging.info("  → Building full OCR text...")
    big_text = build_full_ocr_text(ocr_results)
    save_ocr_debug(ocr_results, big_text, output_dir, base_name)

    # Step 6: Match product types
    logging.info("  → Matching product types from OCR text...")
    matched_product_types = match_product_types_via_lookup(big_text, lookup)

    # Step 7: First split by product type
    logging.info("  → Filtering schema by matched product types...")
    matched, not_matched = split_schema_by_product_type_match(schema, matched_product_types)

    # Step 8: OCR key matching
    logging.info("  → Detecting attribute keys in OCR results...")
    matched_keys, ocr_key_hit = find_key_hits_from_ocr(matched.keys(), ocr_results)
    
    # Step 9: Refine by key hits
    logging.info("  → Refining by detected keys...")
    key_matched, key_not_matched = refine_by_key_hits(matched, not_matched, matched_keys)
    
    # Step 9.5: gen possible values
    logging.info("  → Generating possible values for keys...")
    regions_by_page = detect_table_regions_for_key_hits(ocr_key_hit, images)
    filtered_keys = filter_ocr_keys_by_regions(ocr_key_hit, regions_by_page)
    pages, row_for_key_data = extract_candidate_rows_for_keys(filtered_keys, ocr_results)

    extra_values_dict = {}

    N_CTX = 12288
    MAX_OUTPUT_TOKENS = 4096
    PROMPT_TEMPLATE_TOKENS = 600  # Reserved for system prompt, key, expected_output formatting
    CHARS_PER_TOKEN = 3  # Conservative estimate

    # Derive max OCR chars dynamically from context budget
    _input_token_budget = N_CTX - MAX_OUTPUT_TOKENS - PROMPT_TEMPLATE_TOKENS  # 7592 tokens
    MAX_OCR_CONTEXT_CHARS = _input_token_budget * CHARS_PER_TOKEN 

    for key in key_matched:
        # print(f"\n{'='*60}")
        # print(f"[DEBUG] Processing key: '{key}'")

        expected_output = key_matched[key].get("Expected Output Formatting")
        # print(f"[DEBUG] expected_output: {expected_output}")

        ocr_context_lines = build_ocr_context(ocr_key_hit, key)
        # print(f"[DEBUG] ocr_context_lines ({len(ocr_context_lines)} chars):\n{ocr_context_lines}")

        ocr_context_column = []
        for row in row_for_key_data:
            if row["key"] == key:
                texts = row.get('text')
                for txt in texts:
                    ocr_context_column.append(txt['text'])
        # print(f"[DEBUG] ocr_context_column ({len(ocr_context_column)} items): {ocr_context_column}")

        md_output = format_row_data_to_markdown(ocr_context_column, key)
        # print(f"[DEBUG] md_output:\n{md_output}")

        ocr_context = ocr_context_lines + '\n' + md_output
        # print(f"[DEBUG] ocr_context total length: {len(ocr_context)} chars")

        if len(ocr_context) > MAX_OCR_CONTEXT_CHARS:
            # print(f"[DEBUG] WARNING: context too large ({len(ocr_context)} chars), truncating to {MAX_OCR_CONTEXT_CHARS}")
            logging.warning(
                f"  ⚠️ OCR context for key '{key}' too large ({len(ocr_context)} chars), "
                f"truncating to {MAX_OCR_CONTEXT_CHARS} chars to fit context window."
            )
            ocr_context = ocr_context[:MAX_OCR_CONTEXT_CHARS]

        value_prompt = get_value_prompt(key, expected_output, ocr_context)
        # print(f"[DEBUG] value_prompt:\n{value_prompt}")

        response = generate_llm_response(value_prompt, use_gpu=use_gpu, system_prompt=None)
        # print(f"[DEBUG] raw LLM response (type={type(response).__name__}):\n{response}")

        if response:
            cleaned_response = [item.strip() for item in response.split('\n') if item.strip()]
            # print(f"[DEBUG] cleaned_response: {cleaned_response}")
            extra_values_dict[key] = {
                "possible_extra_values": cleaned_response
            }
        else:
            # print(f"[DEBUG] response was empty/falsy, skipping key '{key}'")
            pass

    # Step 10: Refine by value hits
    logging.info("  → Checking for matching values in text...")
    value_matched, value_not_matched = refine_by_value_hits(key_matched, key_not_matched, big_text, extra_values_dict)

    # Step 11: Refine by key-value pair logic
    logging.info("  → Validating key-value co-occurrence...")
    final_value_matched, final_value_not_matched = refine_by_key_value_pair_matching(value_matched, value_not_matched, big_text)

    # Step 12: Table-based key-value extraction
    logging.info("  → Extracting key-value pairs from detected tables...")
    table_value_matched, table_value_not_matched = match_values_for_keys(
        row_for_key_data,
        value_matched,
        value_not_matched,
        extra_values_dict
    )

    # Step 13: Merge multi-strategy results
    logging.info("  → Merging results from all search strategies...")
    logging.info(table_value_matched)

    final_matched, final_not_matched = merge_match_results(
        (final_value_matched, final_value_not_matched),
        (table_value_matched, table_value_not_matched)
    )

    final_result = final_matched | final_not_matched
    save_final_result(final_result, output_dir=output_dir, base_name=base_name)

    success = bool(final_matched)
    if success:
        logging.info(f"✅ SUCCESS: {len(final_matched)} attribute(s) fully matched in '{base_name}.pdf'")
    else:
        logging.info(f"❌ NO MATCH: No valid key-value pairs found in '{base_name}.pdf'")

    return success
# def process_lighting_spec_sheet(pdf_path, schema_path, ocr_engine, output_dir="final_result", use_gpu=False):
#     base_name = os.path.splitext(os.path.basename(pdf_path))[0]
#     logging.info(f"📄 Processing spec sheet: {base_name}.pdf")

#     # Step 1: PDF to images
#     logging.info("  → Converting PDF to images...")
#     images = convert_pdf_with_pymupdf(pdf_path)

#     # Step 2: OCR
#     logging.info("  → Running OCR on all pages...")
#     ocr_results = get_ocr_object_per_page(images, ocr_engine)

#     # Step 3: Load schema & derive product types
#     logging.info("  → Loading attribute schema and deriving product types...")
#     schema, product_type_set = load_schema_and_derive_product_types(schema_path)

#     llm_output_path = os.path.join(output_dir, "llm_output.json")
#     lookup = {}

#     if os.path.exists(llm_output_path):
#         logging.info(f"  → Found cached LLM output: {llm_output_path}")

#         with open(llm_output_path, "r", encoding="utf-8") as f:
#             lookup = json.load(f)

#         existing_product_types = set(lookup.keys())

#         # Case 1: Perfect match → reuse
#         if existing_product_types == product_type_set:
#             logging.info("  → Cached lookup matches schema product types. Reusing.")
#         else:
#             # Case 2: Partial match → generate missing only
#             missing_product_types = product_type_set - existing_product_types

#             if missing_product_types:
#                 logging.info(
#                     f"  → Generating lookup for missing product types: "
#                     f"{sorted(missing_product_types)}"
#                 )

#                 prompt = build_mounting_prompt(missing_product_types)

#                 new_entries = get_valid_json(
#                     prompt=prompt,
#                     use_gpu=use_gpu
#                 )

#                 if not isinstance(new_entries, dict):
#                     raise ValueError("LLM output must be a dict of product_type -> mounting")

#                 # Append
#                 lookup.update(new_entries)

#                 # Save updated lookup
#                 with open(llm_output_path, "w", encoding="utf-8") as f:
#                     json.dump(lookup, f, indent=2)

#                 logging.info("  → Lookup updated and saved.")
#     else:
#         # Case 3: No cache → generate everything
#         logging.info("  → No cached lookup found. Generating full lookup...")

#         prompt = build_mounting_prompt(product_type_set)

#         lookup = get_valid_json(
#             prompt=prompt,
#             use_gpu=use_gpu
#         )

#         if not isinstance(lookup, dict):
#             raise ValueError("LLM output must be a dict of product_type -> mounting")

#         with open(llm_output_path, "w", encoding="utf-8") as f:
#             json.dump(lookup, f, indent=2)

#         logging.info("  → Full lookup generated and saved.")


#     # Step 5: Build full OCR text
#     logging.info("  → Building full OCR text...")
#     big_text = build_full_ocr_text(ocr_results)

#     # Step 6: Match product types
#     logging.info("  → Matching product types from OCR text...")
#     matched_product_types = match_product_types_via_lookup(big_text, lookup)

#     # Step 7: First split by product type
#     logging.info("  → Filtering schema by matched product types...")
#     matched, not_matched = split_schema_by_product_type_match(schema, matched_product_types)

#     # Step 8: OCR key matching
#     logging.info("  → Detecting attribute keys in OCR results...")
#     matched_keys, ocr_key_hit = find_key_hits_from_ocr(matched.keys(), ocr_results)

#     # Step 9: Refine by key hits
#     logging.info("  → Refining by detected keys...")
#     key_matched, key_not_matched = refine_by_key_hits(matched, not_matched, matched_keys)

#     # >>> NEW: Handle regex guidance caching <<<
#     # Create a stable cache key from the schema file content (or path)
#     with open(schema_path, 'rb') as f:
#         schema_hash = hashlib.md5(f.read()).hexdigest()
#     regex_cache_path = os.path.join(output_dir, f"regex_guidance_{schema_hash}.json")

#     regex_withkey_dict = {}

#     if os.path.exists(regex_cache_path):
#         logging.info("  → Reusing cached regex guidance...")
#         with open(regex_cache_path, "r", encoding="utf-8") as f:
#             regex_withkey_dict = json.load(f)
#     else:
#         logging.info("  → Generating regex guidance from schema (first run)...")
#         guidance = group_schema_by_sentence_closeness(schema)
#         guidance_strip = clean_guidance(guidance)
#         regex_withkey = []

#         for g in guidance_strip:
#             regex_prompt = build_regex_prompt(g)
#             # ⚠️ Fix typo: 'modelq' → 'model'
#             response = generate_llm_response(regex_prompt, use_gpu)
#             response = remove_think_block(response)   
#             regex_withkey.append(response)

#         # Merge responses into one dict
#         regex_withkey_dict = {}
#         for r in regex_withkey:
#             if isinstance(r, str):
#                 try:
#                     parsed = json.loads(r)
#                 except json.JSONDecodeError as e:
#                     logging.warning(f"⚠️ Warning: Failed to parse LLM regex response as JSON: {r[:200]}... Error: {e}")
#                     continue
#                 regex_withkey_dict.update(parsed)
#             elif isinstance(r, dict):
#                 regex_withkey_dict.update(r)
#             else:
#                 raise TypeError(f"Unexpected LLM response type: {type(r)}")

#         # Save to cache
#         os.makedirs(output_dir, exist_ok=True)
#         with open(regex_cache_path, "w", encoding="utf-8") as f:
#             json.dump(regex_withkey_dict, f, indent=2)
#         logging.info(f"  → Regex guidance cached to: {os.path.basename(regex_cache_path)}")
#     # <<< END of regex guidance block >>>    

#     # -----------------------------------------------------------------------------------
#     # Step 10: Refine by value hits
#     logging.info("  → Checking for matching values in text...")
#     value_matched, value_not_matched = refine_by_value_hits(key_matched, key_not_matched, big_text, schema)

#     # Step 11: Refine by key-value pair logic
#     logging.info("  → Validating key-value co-occurrence...")
#     final_value_matched, final_value_not_matched = refine_by_key_value_pair_matching(value_matched, value_not_matched, big_text, regex_withkey_dict)

#     # Step 12: Table-based key-value extraction
#     logging.info("  → Extracting key-value pairs from detected tables...")
#     layout_model = get_yolo_model_path()
#     filter_ocr_key_hit = filter_ocr_key_hit_by_value_matched(ocr_key_hit, value_matched)

#     regions_by_page = detect_table_regions_for_key_hits(filter_ocr_key_hit, ocr_key_hit, value_matched, layout_model, images)

#     filtered_keys = filter_ocr_keys_by_regions(filter_ocr_key_hit, regions_by_page)

#     pages, row_for_key_data = extract_candidate_rows_for_keys(filtered_keys, ocr_results)

#     table_value_matched, table_value_not_matched = match_values_for_keys(
#     row_for_key_data,
#     value_matched,
#     value_not_matched)

#     # Step 13: Merge multi-strategy results
#     logging.info("  → Merging results from all search strategies...")

#     final_matched, final_not_matched = merge_match_results(
#         (final_value_matched, final_value_not_matched),
#         (table_value_matched, table_value_not_matched)
#     )

#     final_result = final_matched | final_not_matched
#     save_final_result(final_result, output_dir=output_dir, base_name=base_name)


#     success = bool(final_value_matched)
#     if success:
#         logging.info(f"✅ SUCCESS: {len(final_value_matched)} attribute(s) fully matched in '{base_name}.pdf'")
#     else:
#         logging.info(f"❌ NO MATCH: No valid key-value pairs found in '{base_name}.pdf'")

#     return success

