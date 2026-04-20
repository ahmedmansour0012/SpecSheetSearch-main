import os
import sys
import ctypes
import logging
import argparse
import glob
import shutil

# ============================================================================
# LOGGING CONFIGURATION (Must be set BEFORE importing llama_cpp modules)
# ============================================================================

def setup_logging(verbose: bool = False):
    """Configure Python logging with optional debug verbosity"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress noisy third-party logs unless in debug mode
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("paddleocr").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"🔧 Logging initialized: {'DEBUG (verbose)' if verbose else 'INFO'} mode")
    return logger

# Check for verbose flag early (before heavy imports)
_VERBOSE_FLAG = "--verbose" in sys.argv or "-v" in sys.argv or os.getenv("VERBOSE", "0") == "1"
logger = setup_logging(verbose=_VERBOSE_FLAG)

# ============================================================================
# THE ATOMIC LLAMA PATCH (PyInstaller compatibility - DO NOT MODIFY)
# ============================================================================
if hasattr(sys, '_MEIPASS'):
    # 1. Locate the bundled library
    bundle_dir = sys._MEIPASS
    lib_path = os.path.join(bundle_dir, "libllama.dylib")
    
    if not os.path.exists(lib_path):
        lib_path = os.path.join(bundle_dir, "llama_cpp", "libllama.dylib")

    # 2. Force load it into the process immediately
    try:
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        logger.debug(f"🔗 Pre-loaded llama library: {lib_path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not pre-load llama library: {e}")

    # 3. MONKEYPATCH the internal loader
    # This prevents the library from ever running its own 'find' logic
    try:
        import llama_cpp._ctypes_extensions as llama_loader
        
        def forced_load_shared_library(*args, **kwargs):
            return ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        
        llama_loader.load_shared_library = forced_load_shared_library
        logger.debug(f"🚀 Llama-cpp loader redirected to: {lib_path}")
    except ImportError as e:
        logger.debug(f"⚠️ Could not patch llama loader (may not be needed): {e}")

# ============================================================================
# IMPORTS (After logging setup and llama patch)
# ============================================================================
from process_lighting_spec_sheet import process_lighting_spec_sheet
from model_loader import get_ocr_instance, get_llm_instance, check_gpu_support

# Suppress mypyc import errors in bundled builds
sys.modules['0deeb2fec52624e647be__mypyc'] = None


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured lighting specs from PDF spec sheets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input ./pdfs --schema schema.json
  %(prog)s --input ./pdfs --schema schema.json --gpu --verbose
  %(prog)s --input ./pdfs --schema schema.json --gpu  # with VERBOSE=1 env var
        """
    )
    parser.add_argument("--gpu", action="store_true", 
                       help="Use GPU acceleration for LLM inference")
    parser.add_argument("--input", required=True, type=str, 
                       help="Path to folder containing input PDF files")
    parser.add_argument("--schema", required=True, type=str, 
                       help="Path to schema JSON file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging (shows llama.cpp internal logs)")
    
    args = parser.parse_args()

    # Re-configure logging if --verbose was passed via argparse
    if args.verbose and not _VERBOSE_FLAG:
        global logger
        logger = setup_logging(verbose=True)
    
    input_pdf_folder = args.input
    schema_path = args.schema

    # Validate inputs
    if not os.path.isdir(input_pdf_folder):
        logger.error(f"❌ Input folder does not exist: {input_pdf_folder}")
        sys.exit(1)
    if not os.path.isfile(schema_path):
        logger.error(f"❌ Schema file not found: {schema_path}")
        sys.exit(1)

    # Find all PDFs (case-insensitive)
    pdf_paths = (
        glob.glob(os.path.join(input_pdf_folder, "*.pdf")) +
        glob.glob(os.path.join(input_pdf_folder, "*.PDF"))
    )

    if not pdf_paths:
        logger.warning(f"⚠️ No PDF files found in {input_pdf_folder}")
        return

    logger.info(f"📄 Found {len(pdf_paths)} PDF(s) to process")
    logger.info(f"🎯 GPU mode: {'ENABLED 🟢' if args.gpu else 'disabled 🔵'}")
    logger.info(f"🔍 Verbose logging: {'ON' if args.verbose else 'off'}")

    # ========================================================================
    # GPU SUPPORT CHECK & MODEL WARMUP
    # ========================================================================
    if args.gpu:
        gpu_supported = check_gpu_support()
        if gpu_supported:
            logger.info("✅ GPU backend detected - llama.cpp can offload layers")
        else:
            logger.warning(
                "⚠️ GPU requested but llama_cpp may not support it.\n"
                "💡 Reinstall with: pip install llama-cpp-python --no-cache-dir -C cmake.args='-DGGML_CUDA=on'"
            )
    
    # Warm up models (OCR + LLM) before processing loop
    try:
        logger.info("🔄 Warming up OCR engine...")
        ocr_engine = get_ocr_instance()
        logger.info("✓ OCR engine ready")
        
        if args.gpu or _VERBOSE_FLAG:
            # Only initialize LLM if GPU mode or verbose (to see loading logs)
            logger.info("🔄 Warming up LLM (this may take 30-60s on first run)...")
            llm = get_llm_instance(use_gpu=args.gpu)
            logger.info("✓ LLM ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}", exc_info=True)
        sys.exit(1)

    # Define output structure
    output_dir = "final_result"
    success_dir = os.path.join(output_dir, "success_found")
    not_found_dir = os.path.join(output_dir, "not_found")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(not_found_dir, exist_ok=True)

    logger.info(f"📁 Output directories ready: {output_dir}")

    # ========================================================================
    # PROCESSING LOOP
    # ========================================================================
    success_count = 0
    fail_count = 0
    
    for idx, pdf_path in enumerate(pdf_paths, 1):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"📄 [{idx}/{len(pdf_paths)}] Processing: {os.path.basename(pdf_path)}")
            logger.info(f"📍 Path: {pdf_path}")
            
            is_hit = process_lighting_spec_sheet(
                pdf_path,
                schema_path,
                ocr_engine,
                output_dir=output_dir,
                use_gpu=args.gpu
            )

            filename = os.path.basename(pdf_path)
            if is_hit:
                dest = os.path.join(success_dir, filename)
                logger.info(f"✅ Success: moving {filename} to success folder")
                success_count += 1
            else:
                dest = os.path.join(not_found_dir, filename)
                logger.info(f"❌ No match: moving {filename} to not_found folder")
                fail_count += 1

            # Move original PDF
            shutil.move(pdf_path, dest)
            logger.debug(f"📦 Moved: {pdf_path} → {dest}")

        except Exception as e:
            logger.error(f"⚠️ Error processing {pdf_path}: {e}", exc_info=args.verbose)
            filename = os.path.basename(pdf_path)
            dest = os.path.join(not_found_dir, filename)
            logger.info(f"📁 Moving failed PDF {filename} to not_found folder")
            try:
                shutil.move(pdf_path, dest)
            except Exception as move_err:
                logger.error(f"❌ Could not move failed file: {move_err}")
            fail_count += 1

    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("✨ Processing complete!")
    logger.info(f"📊 Results: {success_count} succeeded, {fail_count} failed")
    logger.info(f"📁 Success files: {success_dir}")
    logger.info(f"📁 Failed files: {not_found_dir}")
    
    if args.gpu and _VERBOSE_FLAG:
        logger.info("💡 Tip: Monitor GPU usage during processing with:")
        logger.info("   • NVIDIA: watch -n 1 nvidia-smi")
        logger.info("   • Mac:   powermetrics --samplers gpu_power -i1000")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Unhandled exception: {e}", exc_info=True)
        sys.exit(1)