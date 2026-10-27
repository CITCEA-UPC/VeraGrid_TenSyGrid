import asyncio
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="TenSyGrid Stability Monitor")
app.mount("/resources", StaticFiles(directory=Path(__file__).parent / "resources"), name="resources")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TENSYGRID_DIR = Path(__file__).resolve().parents[1]
GRIDS_DIR = PROJECT_ROOT / "Grids_and_profiles" / "grids"
PRECOMPUTED_DIR = TENSYGRID_DIR / "precomputed_builds"
SCRIPT_PATH = TENSYGRID_DIR / "small_signal_analysis_main.py"
EMT_SCRIPT_PATH = TENSYGRID_DIR / "small_signal_analysis_main_emt.py"
CONTINGENCY_SCRIPT_PATH = TENSYGRID_DIR / "run_contingency.py"


def _find_venv_python() -> Path:
    """Find the venv Python executable, searching several candidate locations.

    Handles Windows (Scripts/python.exe) and Unix (bin/python3 or bin/python),
    and tolerates the venv being one directory above or below PROJECT_ROOT.
    Falls back to the current interpreter if nothing is found.
    """
    # Candidate venv root directories (the repo root, one level up, one level down)
    search_roots = [
        PROJECT_ROOT,
        PROJECT_ROOT.parent,
        PROJECT_ROOT / "venv",   # venv directly as a subfolder named venv
        PROJECT_ROOT / ".venv",  # .venv directly as a subfolder named .venv
    ]
    # Also add every direct child dir of PROJECT_ROOT named *venv* or *.venv*
    try:
        for child in PROJECT_ROOT.iterdir():
            if child.is_dir() and "venv" in child.name.lower():
                search_roots.append(child)
    except Exception:
        pass

    # Relative paths inside a venv root that point to the interpreter
    candidates_rel = [
        Path("venv") / "Scripts" / "python.exe",   # Windows
        Path("venv") / "bin" / "python3",           # Linux / macOS
        Path("venv") / "bin" / "python",            # fallback
        Path("Scripts") / "python.exe",             # if root IS the venv
        Path("bin") / "python3",
        Path("bin") / "python",
    ]

    for root in search_roots:
        for rel in candidates_rel:
            candidate = root / rel
            if candidate.exists():
                return candidate

    # Last resort: use the interpreter that is running this script
    return Path(sys.executable)


VENV_PYTHON = _find_venv_python()

_running_process: Optional[subprocess.Popen] = None
_current_grid: Optional[str] = None
_running_logs: Optional[tuple[Path, Path]] = None

_contingency_process: Optional[subprocess.Popen] = None
_contingency_info: Optional[str] = None
_contingency_logs: Optional[tuple[Path, Path]] = None


class RunRequest(BaseModel):
    grid_filename: str
    study_type: str = "RMS"


class ContingencyRequest(BaseModel):
    grid_filename: str
    contingency_type: str
    contingency_element: str
    study_type: str = "RMS"


def _new_log_paths(prefix: str) -> tuple[Path, Path]:
    """Create a pair of temporary log files for a child process.

    Child output must never go through an unread ``subprocess.PIPE``: if the
    process writes more than the OS pipe buffer, it blocks before finishing and
    the parent waits for it forever. Temporary files avoid that deadlock.
    """
    log_dir = Path(tempfile.gettempdir()) / "tensygrid_dashboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{os.getpid()}_{time.time_ns()}"
    return (
        log_dir / f"{prefix}_{stamp}_stdout.log",
        log_dir / f"{prefix}_{stamp}_stderr.log",
    )


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_process_logs(logs: Optional[tuple[Path, Path]]) -> tuple[str, str]:
    if logs is None:
        return "", ""
    stdout = _read_log(logs[0])
    stderr = _read_log(logs[1])
    _remove_logs(logs)
    return stdout, stderr


def _remove_logs(logs: Optional[tuple[Path, Path]]) -> None:
    if logs is None:
        return
    for path in logs:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _spawn_logged(command: list[str],
                  prefix: str,
                  cwd: Path,
                  env: Optional[dict[str, str]] = None) -> tuple[subprocess.Popen, tuple[Path, Path]]:
    """Start a child process with its output redirected to temporary files."""
    logs = _new_log_paths(prefix)
    with open(logs[0], "w", encoding="utf-8") as stdout_file, \
            open(logs[1], "w", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
    return process, logs


def _infer_study_type(data: dict, filename: str) -> str:
    """Read the study type, falling back to the result filename for old files."""
    study = data.get("study")
    if study:
        return str(study).upper()
    return "EMT" if "_emt_" in Path(filename).stem else "RMS"


def get_available_grids() -> list[dict]:
    grids = []
    if not GRIDS_DIR.exists():
        return grids
    
    for f in sorted(GRIDS_DIR.iterdir()):
        if f.suffix in ('.gridcal', '.veragrid'):
            grids.append({
                'filename': f.name,
                'name': f.name,
                'size_kb': round(f.stat().st_size / 1024, 1),
            })
    return grids


def get_precomputed_results() -> list[dict]:
    results = []
    if not PRECOMPUTED_DIR.exists():
        return results
    
    for f in sorted(PRECOMPUTED_DIR.iterdir()):
        if f.suffix == '.tensygrid':
            try:
                with open(f, 'rb') as fp:
                    data = pickle.load(fp)
                results.append({
                    'filename': f.name,
                    'grid_name': data.get('grid_name', f.stem),
                    'study': _infer_study_type(data, f.name),
                    'stable': data.get('stable'),
                    'margin': data.get('margin'),
                    'n_eigenvalues': data.get('n_eigenvalues', 0),
                    'modified': f.stat().st_mtime,
                    'elapsed_seconds': data.get('elapsed_seconds'),
                })
            except Exception as e:
                results.append({
                    'filename': f.name,
                    'grid_name': f.stem,
                    'study': 'RMS',
                    'error': str(e),
                    'modified': f.stat().st_mtime,
                })
    results.sort(key=lambda r: r.get('modified', 0.0), reverse=True)
    return results


def load_result(filename: str) -> dict:
    filepath = PRECOMPUTED_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        result = {
            'grid_name': data.get('grid_name', 'Unknown'),
            'study': _infer_study_type(data, filename),
            'stable': data.get('stable', False),
            'margin': data.get('margin', 0.0),
            'margin_definition': data.get('margin_definition'),
            'unstable_threshold': data.get('unstable_threshold'),
            'period': data.get('period'),
            'n_eigenvalues': data.get('n_eigenvalues', 0),
            'n_state_vars': data.get('n_state_vars', 0),
            'eigenvalues': data.get('eigenvalues', []),
            'critical_modes': data.get('critical_modes', []),
            'bus_voltages': data.get('bus_voltages', []),
            'grid_info': data.get('grid_info', {}),
            'matches': data.get('matches', 0),
            'op_hash': data.get('op_hash', ''),
            'timestamp': filepath.stat().st_mtime,
            'elapsed_seconds': data.get('elapsed_seconds'),
        }
        
        if result['stable']:
            result['status'] = 'STABLE'
        elif result['margin'] < 0.001:
            result['status'] = 'MARGINAL'
        else:
            result['status'] = 'UNSTABLE'
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading result: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/grids")
async def list_grids():
    return get_available_grids()


@app.get("/api/results")
async def list_results():
    return get_precomputed_results()


@app.get("/api/results/{filename}")
async def get_result(filename: str):
    return load_result(filename)


@app.post("/api/run")
async def run_simulation(request: RunRequest):
    global _running_process, _current_grid, _running_logs

    if _running_process is not None and _running_process.poll() is None:
        raise HTTPException(
            status_code=409, 
            detail=f"Simulation already running for grid: {_current_grid}"
        )
    
    grid_path = GRIDS_DIR / request.grid_filename
    if not grid_path.exists():
        raise HTTPException(status_code=404, detail="Grid file not found")
    
    _current_grid = request.grid_filename

    script_path = EMT_SCRIPT_PATH if request.study_type.upper() == "EMT" else SCRIPT_PATH

    _remove_logs(_running_logs)
    _running_process, _running_logs = _spawn_logged(
        [str(VENV_PYTHON), str(script_path), request.grid_filename],
        prefix="simulation",
        cwd=TENSYGRID_DIR,
    )

    return {
        "status": "started",
        "grid": request.grid_filename,
        "pid": _running_process.pid,
    }


@app.get("/api/status")
async def get_status():
    global _running_process, _current_grid, _running_logs

    if _running_process is None:
        return {"status": "idle", "grid": None}
    
    poll = _running_process.poll()
    
    if poll is None:
        return {
            "status": "running",
            "grid": _current_grid,
            "pid": _running_process.pid,
        }

    stdout, stderr = _read_process_logs(_running_logs)
    _running_logs = None

    result = {
        "status": "completed" if poll == 0 else "failed",
        "grid": _current_grid,
        "returncode": poll,
        "stdout": stdout[-2000:] if stdout else "",
        "stderr": stderr[-2000:] if stderr else "",
    }
    
    _running_process = None
    _current_grid = None
    
    return result


@app.post("/api/cancel")
async def cancel_simulation():
    global _running_process, _current_grid, _running_logs

    if _running_process is None or _running_process.poll() is not None:
        raise HTTPException(status_code=404, detail="No simulation running")

    _running_process.terminate()
    try:
        _running_process.wait(timeout=5)
    except Exception:
        _running_process.kill()

    grid = _current_grid
    _running_process = None
    _current_grid = None
    _remove_logs(_running_logs)
    _running_logs = None
    return {"status": "cancelled", "grid": grid}


@app.post("/api/cancel-contingency")
async def cancel_contingency():
    global _contingency_process, _contingency_info, _contingency_logs

    if _contingency_process is None or _contingency_process.poll() is not None:
        raise HTTPException(status_code=404, detail="No contingency running")

    _contingency_process.terminate()
    try:
        _contingency_process.wait(timeout=5)
    except Exception:
        _contingency_process.kill()

    info = _contingency_info
    _contingency_process = None
    _contingency_info = None
    _remove_logs(_contingency_logs)
    _contingency_logs = None
    return {"status": "cancelled", "info": info}



@app.delete("/api/results/{filename}")
async def delete_result(filename: str):
    filepath = PRECOMPUTED_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    filepath.unlink()
    return {"status": "deleted", "filename": filename}


@app.post("/api/run-contingency")
async def run_contingency(request: ContingencyRequest):
    global _contingency_process, _contingency_info, _contingency_logs
    
    if _contingency_process is not None and _contingency_process.poll() is None:
        raise HTTPException(
            status_code=409,
            detail=f"Contingency already running: {_contingency_info}"
        )
    
    grid_path = GRIDS_DIR / request.grid_filename
    if not grid_path.exists():
        raise HTTPException(status_code=404, detail="Grid file not found")
    
    _contingency_info = f"{request.contingency_type}:{request.contingency_element} on {request.grid_filename}"
    
    _remove_logs(_contingency_logs)
    _contingency_process, _contingency_logs = _spawn_logged(
        [
            str(VENV_PYTHON),
            str(CONTINGENCY_SCRIPT_PATH),
            "--grid", request.grid_filename,
            "--type", request.contingency_type,
            "--element", request.contingency_element,
            "--study", request.study_type.upper(),
        ],
        prefix="contingency",
        cwd=TENSYGRID_DIR,
    )

    return {
        "status": "started",
        "info": _contingency_info,
        "pid": _contingency_process.pid,
    }


@app.get("/api/contingency-status")
async def get_contingency_status():
    global _contingency_process, _contingency_info, _contingency_logs

    if _contingency_process is None:
        return {"status": "idle", "info": None}
    
    poll = _contingency_process.poll()
    
    if poll is None:
        return {
            "status": "running",
            "info": _contingency_info,
            "pid": _contingency_process.pid,
        }

    stdout, stderr = _read_process_logs(_contingency_logs)
    _contingency_logs = None

    if poll == 0:
        try:
            data = None
            combined = (stdout or "") + "\n" + (stderr or "")
            # Try greedy from end to find largest JSON object
            for m in re.finditer(r'\{.*\}', combined, re.DOTALL):
                try:
                    candidate = m.group(0)
                    data = json.loads(candidate)
                    if isinstance(data, dict) and ("grid_name" in data or "error" in data or "status" in data):
                        break
                except Exception:
                    data = None
                    continue
            if data is None:
                raise json.JSONDecodeError("No JSON found", "", 0)
            result = {
                "status": "completed",
                "info": _contingency_info,
                "returncode": poll,
                "data": data,
            }
        except json.JSONDecodeError:
            result = {
                "status": "failed",
                "info": _contingency_info,
                "returncode": poll,
                "stdout": stdout[-2000:] if stdout else "",
                "stderr": "Failed to parse results: " + stderr[-1000:] if stderr else "Failed to parse results",
            }
    else:
        result = {
            "status": "failed",
            "info": _contingency_info,
            "returncode": poll,
            "stdout": stdout[-2000:] if stdout else "",
            "stderr": stderr[-2000:] if stderr else "",
        }
    
    _contingency_process = None
    _contingency_info = None
    
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8550)
