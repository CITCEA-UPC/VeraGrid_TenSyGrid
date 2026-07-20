import asyncio
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

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

_contingency_process: Optional[subprocess.Popen] = None
_contingency_info: Optional[str] = None


class RunRequest(BaseModel):
    grid_filename: str


class ContingencyRequest(BaseModel):
    grid_filename: str
    contingency_type: str
    contingency_element: str


def get_available_grids() -> list[dict]:
    grids = []
    if not GRIDS_DIR.exists():
        return grids
    
    for f in sorted(GRIDS_DIR.iterdir()):
        if f.suffix == '.gridcal':
            grids.append({
                'filename': f.name,
                'name': f.stem,
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
                    'error': str(e),
                    'modified': f.stat().st_mtime,
                })
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
            'stable': data.get('stable', False),
            'margin': data.get('margin', 0.0),
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
    return HTMLResponse(html_path.read_text())


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
    global _running_process, _current_grid
    
    if _running_process is not None and _running_process.poll() is None:
        raise HTTPException(
            status_code=409, 
            detail=f"Simulation already running for grid: {_current_grid}"
        )
    
    grid_path = GRIDS_DIR / request.grid_filename
    if not grid_path.exists():
        raise HTTPException(status_code=404, detail="Grid file not found")
    
    _current_grid = request.grid_filename
    
    _running_process = subprocess.Popen(
        [str(VENV_PYTHON), str(SCRIPT_PATH), request.grid_filename],
        cwd=str(TENSYGRID_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    return {
        "status": "started",
        "grid": request.grid_filename,
        "pid": _running_process.pid,
    }


@app.get("/api/status")
async def get_status():
    global _running_process, _current_grid
    
    if _running_process is None:
        return {"status": "idle", "grid": None}
    
    poll = _running_process.poll()
    
    if poll is None:
        return {
            "status": "running",
            "grid": _current_grid,
            "pid": _running_process.pid,
        }
    
    stdout = _running_process.stdout.read() if _running_process.stdout else ""
    stderr = _running_process.stderr.read() if _running_process.stderr else ""
    
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
    global _running_process, _current_grid

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
    return {"status": "cancelled", "grid": grid}


@app.post("/api/cancel-contingency")
async def cancel_contingency():
    global _contingency_process, _contingency_info

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
    global _contingency_process, _contingency_info
    
    if _contingency_process is not None and _contingency_process.poll() is None:
        raise HTTPException(
            status_code=409,
            detail=f"Contingency already running: {_contingency_info}"
        )
    
    grid_path = GRIDS_DIR / request.grid_filename
    if not grid_path.exists():
        raise HTTPException(status_code=404, detail="Grid file not found")
    
    _contingency_info = f"{request.contingency_type}:{request.contingency_element} on {request.grid_filename}"
    
    _contingency_process = subprocess.Popen(
        [
            str(VENV_PYTHON),
            str(CONTINGENCY_SCRIPT_PATH),
            "--grid", request.grid_filename,
            "--type", request.contingency_type,
            "--element", request.contingency_element,
        ],
        cwd=str(TENSYGRID_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    return {
        "status": "started",
        "info": _contingency_info,
        "pid": _contingency_process.pid,
    }


@app.get("/api/contingency-status")
async def get_contingency_status():
    global _contingency_process, _contingency_info
    
    if _contingency_process is None:
        return {"status": "idle", "info": None}
    
    poll = _contingency_process.poll()
    
    if poll is None:
        return {
            "status": "running",
            "info": _contingency_info,
            "pid": _contingency_process.pid,
        }
    
    stdout = _contingency_process.stdout.read() if _contingency_process.stdout else ""
    stderr = _contingency_process.stderr.read() if _contingency_process.stderr else ""
    
    if poll == 0:
        try:
            data = json.loads(stdout)
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
