#!/usr/bin/env python3
import os
import sys
import subprocess
import importlib.metadata
import glob
from pathlib import Path
import doclayout_yolo
import llama_cpp
import paddle

OUTPUT_NAME = "keySearch"
SEP = os.pathsep 

# 1️⃣ Verification
if not Path("app/main.py").exists():
    print("❌ main.py not found. Run from project root.")
    sys.exit(1)

# 2️⃣ Gather Pathing Info
doclayout_path = os.path.dirname(doclayout_yolo.__file__)
paddle_path = os.path.dirname(paddle.__file__)
llama_path = os.path.dirname(llama_cpp.__file__)
llama_cpp_dir = str(Path(llama_cpp.__file__).parent)
# 3️⃣ Define Data Folders
# We include the entire doclayout_yolo folder to ensure cfg/default.yaml is present
data_folders = [
    ("app/*", "app"),
    (doclayout_path, "doclayout_yolo"), # Crucial: Maps full package to bundle
    (paddle_path, "paddle"),
    (llama_path, "llama_cpp"),
]
# Specifically for llama-cpp-python, we often need to add the binaries directly
# find all .dylib, .so, or .dll files in that folder
for file in os.listdir(llama_path):
    if file.endswith(('.dylib', '.so', '.dll')):
        data_folders.append((os.path.join(llama_path, file), "."))

# 4️⃣ Define Collections & Metadata
collect_all = [
    "paddleocr", "paddle", "paddlex",
    "pypdfium2", "PyMuPDF", "langchain", "tqdm", "regex",
    "PyPDF2", "dill", "hf_xet", "tiktoken", "pillow",
    "cv2", "opencv-contrib-python"
]

metadata_to_copy = [
    "paddlex", "paddleocr", "opencv-contrib-python", 
    "shapely", "pyclipper", "lanms-neo"
]

hidden_imports = [
    "cv2", "paddlex.inference.utils.io.readers",
    "huggingface_hub", "llama_cpp", "shapely", "pyclipper", "lanms_neo",
    "paddlex.inference.models.ocr", "paddlex.inference.pipelines.ocr",
    "paddle.base.libpaddle", "paddle.utils.cpp_extension"
]

# 5️⃣ Construct Command
cmd_parts = [
    "pyinstaller", "--onefile", "--console", "--clean",
    "--distpath=dist", "--workpath=build", "--specpath=.",
    "--name=" + OUTPUT_NAME
]
# Add this to your "hidden_imports" section if not there
hidden_imports += [
    "paddle.libs", 
    "paddle.base.core",
    "paddle.jit"
]

# Add this to your "cmd_parts" to ensure it grabs the dynamic libs
cmd_parts.append("--collect-submodules=paddle")
# 1. Path to the package
llama_dir = os.path.dirname(llama_cpp.__file__)

# 2. Find all dylibs
# dylibs = glob.glob(os.path.join(llama_dir, "*.dylib"))
# Find all dynamic libraries
libs = glob.glob(os.path.join(llama_dir, "*.dylib")) + \
       glob.glob(os.path.join(llama_dir, "*.so")) + \
       glob.glob(os.path.join(llama_dir, "*.dll"))

for lib in libs:
    cmd_parts.append(f"--add-binary={lib}{SEP}.")
    cmd_parts.append(f"--add-binary={lib}{SEP}llama_cpp")
    print(f"💎 Linking binary: {os.path.basename(lib)}")

    
# Add Metadata
for pkg in metadata_to_copy:
    try:
        importlib.metadata.distribution(pkg)
        cmd_parts.append(f"--copy-metadata={pkg}")
    except importlib.metadata.PackageNotFoundError:
        print(f"⚠️ Metadata for '{pkg}' not found.")

# Add Hidden Imports
for mod in hidden_imports:
    cmd_parts.append(f"--hidden-import={mod}")

# Add Full Collections
for pkg in collect_all:
    cmd_parts.append(f"--collect-all={pkg}")

# Add Data Folders (This now includes doclayout_yolo/cfg correctly)
for src, dest in data_folders:
    cmd_parts.append(f"--add-data={src}{SEP}{dest}")

# Add Runtime Hooks
if Path("hooks/rthook_paddle.py").exists():
    cmd_parts.append("--runtime-hook=hooks/rthook_paddle.py")
if Path("hooks").exists():
    cmd_parts.append("--additional-hooks-dir=hooks")

cmd_parts.append("--collect-data=paddlex")
cmd_parts.append("app/main.py")

# 6️⃣ Execution
full_cmd = " ".join(cmd_parts)
print(f"\n🚀 Executing PyInstaller Build for {OUTPUT_NAME}...\n")
result = subprocess.run(full_cmd, shell=True)

if result.returncode == 0:
    print(f"\n✅ Build Successful! Check dist/{OUTPUT_NAME}")
else:
    print(f"\n❌ Build Failed with code {result.returncode}")
    