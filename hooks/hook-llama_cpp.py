# hooks/hook-llama_cpp.py
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# This tells PyInstaller to grab every binary and data file in the package
datas = collect_data_files('llama_cpp')
binaries = collect_dynamic_libs('llama_cpp')