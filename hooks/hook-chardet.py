# hooks/hook-chardet.py
from PyInstaller.utils.hooks import collect_submodules

# Exclude the problematic mypyc compiled module
hiddenimports = [m for m in collect_submodules("chardet") if "__mypyc" not in m]
