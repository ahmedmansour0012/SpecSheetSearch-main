# SpecSheetSearch

---

### 🔧 Installation Instructions

1. **Clone the repository**
```bash
git clone <repo-url>
cd SpecSheetSearch
```

2. **Create a virtual environment**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install `llama-cpp-python`**

> **CPU-only (recommended for most users):**
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

> **GPU (CUDA):**
```bash
# Linux/macOS
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir

# Windows (Command Prompt)
set CMAKE_ARGS=-DGGML_CUDA=on
set FORCE_CMAKE=1
pip install llama-cpp-python --no-cache-dir
```
---

### ▶️ Usage

1. Place your lighting spec sheets in the input folder:
```bash
mkdir -p data/new_pdfs
cp *.pdf data/new_pdfs/
```

2. Run the extractor:
```bash
# Basic usage
python main.py --input ./data/new_pdfs --schema ./schema/lighting_schema.json

# With GPU (if supported)
python app/main.py --gpu --input ./data/new_pdfs --schema ./schema/lighting_schema.json
```

✅ **Results:**
- Successfully extracted specs → `final_result/success_found/`
- No-match PDFs → `final_result/not_found/`
