# Local Build Instructions (macOS & Ubuntu)

These instructions guide you to build the `keySearch` executable **locally** on your machine without GitHub Actions.  

---

## 1. Prerequisites

- **Python 3.12**  
- **Homebrew** (macOS) or **apt** (Ubuntu) for system dependencies  
- `git` to clone the repository  

---

## 2. Clone Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
````

---

## 3. Install System Dependencies

### macOS

```bash
brew update
brew install cmake ninja libomp geos ccache
```

### Ubuntu

```bash
sudo apt update
sudo apt install -y cmake ninja-build libomp-dev libgeos-dev build-essential ccache
```

---

## 4. Set Up Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

---

## 5. Install Python Dependencies

```bash
pip install "paddlex[ocr]" "paddlex[ocr-core]"
pip install shapely pyclipper lanms-neo
pip install pyinstaller
pip uninstall opencv-python -y || true
pip install opencv-contrib-python
pip install "llama-cpp-python==0.3.15"
pip install -r requirements.txt
```

---

## 6. Fix PaddlePaddle Binary Linkage

```bash
CORE_PATH="venv/lib/python3.12/site-packages/paddle/base/core.py"
rm -f "$CORE_PATH"
cp core.py "$CORE_PATH"
# pip install "llama-cpp-python==0.3.15"
```

> This replaces `core.py` with a custom version to fix binary linkage issues.

---

## 7. Set Build Environment

```bash
export DYLD_LIBRARY_PATH=$(python -c "import os, llama_cpp; print(os.path.dirname(llama_cpp.__file__))")
```

> For Linux, this step is usually not necessary.

---

## 8. Build Executable

```bash
python build.py
```

This generates a standalone executable in the `dist/` directory. Ensure `build.py` is configured to collect all required PaddleX modules and hidden imports.


✅ **Tip:** Keep your virtual environment active whenever you rebuild to ensure all dependencies are linked correctly.
