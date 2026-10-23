"""
DermatoAI - Root Entrypoint Wrapper
Mengarahkan eksekusi ke app/app.py untuk kompatibilitas penuh dengan Streamlit Cloud deployment.
"""
from pathlib import Path
import sys

_app_dir = Path(__file__).resolve().parent / "app"
_target_script = _app_dir / "app.py"

if _target_script.exists():
    sys.path.insert(0, str(_app_dir))
    with open(_target_script, encoding="utf-8") as _f:
        exec(compile(_f.read(), str(_target_script), "exec"), globals())
else:
    raise FileNotFoundError(f"File aplikasi Streamlit tidak ditemukan pada {_target_script}")
