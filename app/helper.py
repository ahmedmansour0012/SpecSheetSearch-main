import logging
import itertools

def generate_ocr_variants(term, max_variants=10000):
    """
    Generates common OCR-friendly variants for a string/number.
    Example: '109' -> ['109', '1o9', 'l09', 'lo9']

    If the number of combinations would exceed max_variants, each ambiguous
    character is limited to its 2 most likely substitutions (original + top
    substitute) to keep the result set manageable.
    """
    term = str(term)

    # All values are lowercase — no uppercase needed since we lowercase everything
    ocr_map = {
        '0': ['0', 'o'],
        '1': ['1', 'l', 'i'],
        'i': ['i', 'l', '1'],
        'l': ['l', 'i', '1'],
        '5': ['5', 's'],
        '8': ['8', 'b'],
        '9': ['9', 'g', 'q'],
        '2': ['2', 'z'],
        '6': ['6', 'g', 'b'],
    }

    # Build options per character, deduplicating within each position
    chars_options = []
    for c in term.lower():
        seen = set()
        opts = [r for r in ocr_map.get(c, [c]) if not (r in seen or seen.add(r))]
        chars_options.append(opts)

    # Compute total combinations
    total_combos = 1
    for opts in chars_options:
        total_combos *= len(opts)

    # If explosion, trim each position to at most 2 options (original + top substitute)
    if total_combos > max_variants:
        chars_options = [opts[:2] for opts in chars_options]
        reduced = 1
        for opts in chars_options:
            reduced *= len(opts)
        logging.debug(
            f"OCR variants for '{term}' capped: {total_combos} → {reduced} combinations."
        )

    variants = sorted({''.join(combo) for combo in itertools.product(*chars_options)})
    logging.debug(f"Generated {len(variants)} OCR variant(s) for term '{term}': {variants}")
    return variants
