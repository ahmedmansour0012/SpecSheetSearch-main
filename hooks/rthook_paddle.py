import sys
import os

if hasattr(sys, '_MEIPASS'):
    bundle_dir = sys._MEIPASS
    
    # This is the directory PyInstaller creates for your data_folders
    # Mapping doclayout_path -> 'doclayout_yolo' in build.py results in this:
    yolo_cfg_dir = os.path.join(bundle_dir, "doclayout_yolo", "cfg")
    
    # Force the library to look HERE for its YAML files
    os.environ['YOLO_CONFIG_DIR'] = yolo_cfg_dir
    # Some versions use this specific env var for the default config
    os.environ['ULTRALYTICS_CONFIG_DIR'] = yolo_cfg_dir
    # Also add to PATH just in case
    os.environ['PATH'] = bundle_dir + os.pathsep + os.environ.get('PATH', '')
    os.environ['LLAMA_CPP_LIB'] = os.path.join(bundle_dir, "llama_cpp")

# if sys.platform.startswith('linux'):
#     # sys._MEIPASS is the path to the temporary folder where the app is unpacked
#     meipass = getattr(sys, '_MEIPASS', None)
#     if meipass:
#         paddle_libs = os.path.join(meipass, 'paddle', 'libs')
#         # Prepend paddle libs to the environment path
#         os.environ['LD_LIBRARY_PATH'] = paddle_libs + ":" + os.environ.get('LD_LIBRARY_PATH', '')
        
#         print(f"🚀 PaddleX Dependency Checks Bypassed")
#         print(f"✅ LD_LIBRARY_PATH set to: {paddle_libs}")
# if sys.platform.startswith("darwin"):
#     meipass = getattr(sys, '_MEIPASS', None)
#     if meipass:
#         paddle_libs = os.path.join(meipass, 'paddle', 'libs')
#         # Prepend paddle libs to the environment path
#         os.environ['DYLD_LIBRARY_PATH'] = paddle_libs + ":" + os.environ.get('DYLD_LIBRARY_PATH', '')
        
#         print(f"🚀 PaddleX Dependency Checks Bypassed")
#         print(f"✅ DYLD_LIBRARY_PATH set to: {paddle_libs}")
    
if hasattr(sys, "_MEIPASS"):
    meipass = sys._MEIPASS
    paddle_libs = os.path.join(meipass, "paddle", "libs")

    if sys.platform.startswith("linux"):
        os.environ["LD_LIBRARY_PATH"] = paddle_libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        print(f"LD_LIBRARY_PATH set to: {paddle_libs}")

    elif sys.platform.startswith('darwin'):
        os.environ["DYLD_LIBRARY_PATH"] = paddle_libs + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
        print(f"DYLD_LIBRARY_PATH set to: {paddle_libs}")
# --- The Monkeypatch to bypass PaddleX checks ---
try:
    import paddlex.utils.deps as paddlex_deps
    def mock_true(*args, **kwargs): return True
    paddlex_deps.require_extra = mock_true
    paddlex_deps.require_deps = mock_true
    paddlex_deps.require_all_deps = mock_true
    print("🚀 PaddleX Dependency Checks Bypassed")
except Exception:
    pass