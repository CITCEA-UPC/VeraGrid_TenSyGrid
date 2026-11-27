# Installation and GUI startup

This guide takes you from installing the tools to opening the **TenSyGrid Stability Monitor**, the browser-based GUI included in this repository.

## 1. Install Git and Python

Skip this step if both tools are already installed. Python **3.12** is the version used in the repository's current local environment; use it for the commands below. The dependency versions in `requirements.txt` also affect which Python versions can be used.

### Windows

1. Download and install [Git for Windows](https://git-scm.com/install/windows). Keep the option that makes Git available from the command line.
2. Install Python 3.12 using the [official Python downloads](https://www.python.org/downloads/). If the installer offers **Add Python to PATH**, enable it.
3. Open a new **PowerShell** window and check:

```powershell
git --version
py -3.12 --version
```

Use PowerShell for the Windows commands in this guide.

### Ubuntu / Debian

Install Git, Python and the virtual environment tools:

```bash
sudo apt update
sudo apt install git python3 python3-venv python3-pip
git --version
python3 --version
```

The Python version provided depends on your distribution. If it is not 3.12, install Python 3.12 alongside the system Python using the tools available for your distribution, and use `python3.12` instead of `python3` when creating the virtual environment below. Do not replace the system Python.

### macOS

Install Git using the [official Git installation instructions](https://git-scm.com/install/mac) and Python 3.12 from the [official Python downloads](https://www.python.org/downloads/macos/). Open **Terminal** and check:

```bash
git --version
python3.12 --version
```

Use `python3.12` instead of `python3` in the environment creation command below if another Python version is also installed.

## 2. Clone the repository

In your terminal, move to the folder where you want to keep the project, then run:

```bash
git clone https://github.com/CITCEA-UPC/VeraGrid_TenSyGrid.git
cd VeraGrid_TenSyGrid
```

If GitHub asks you to sign in, use an account with access to the repository. Run all subsequent commands from this repository folder, which contains `README.md` and `requirements.txt`.

## 3. Create and activate a virtual environment

A virtual environment keeps the project's Python dependencies separate from other projects. Create it once; activate it whenever you open a new terminal to work on the project. See the [Python virtual environment documentation](https://docs.python.org/3.12/library/venv.html) for more details.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

The remaining Windows examples use PowerShell syntax, so return to PowerShell and use `.\.venv\Scripts\python.exe` instead of `python` if activation is blocked there. Activation is optional when you use the environment's Python executable directly.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, check that Python points inside the project's `.venv` folder:

```bash
python -c "import sys; print(sys.executable)"
```

## 4. Install the dependencies

With the virtual environment active, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r src/TensyGridEngine/gui/requirements.txt matplotlib
```

This installs the main requirements, the web GUI requirements and Matplotlib, which the stability analysis scripts import. The first installation can take several minutes and needs an internet connection. Wait for it to finish successfully before starting the GUI.

If pip reports **No matching distribution found**, check the named package, your Python version and whether your package index provides the version requested in `requirements.txt`.

## 5. Start the GUI and open it in a browser

Keep the virtual environment active and run the command for your operating system from the repository root. Setting `PYTHONPATH` makes the code in `src/` available to Python.

### Windows (PowerShell)

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m uvicorn TensyGridEngine.gui.dashboard:app --host 127.0.0.1 --port 8550
```

### Linux / macOS

```bash
PYTHONPATH=src python -m uvicorn TensyGridEngine.gui.dashboard:app --host 127.0.0.1 --port 8550
```

When the terminal reports that the server has started, open **[http://localhost:8550](http://localhost:8550)** in your browser. The equivalent address is [http://127.0.0.1:8550](http://127.0.0.1:8550).

Keep the terminal open while using the GUI. Select one of the bundled example grids in the Stability Monitor to start exploring. Press **Ctrl+C** in the terminal to stop the server.

## 6. If the port is already in use

The default port is **8550**. Always check the server URL printed in the terminal: if you launch the GUI on another port, open that port in your browser too.

The Uvicorn commands above explicitly select port 8550. If you see **address already in use** or **WinError 10048**, rerun the command with `--port 8551` instead of `--port 8550` (or try 8552 if necessary):

**Windows (PowerShell), after setting `PYTHONPATH` as above:**

```powershell
python -m uvicorn TensyGridEngine.gui.dashboard:app --host 127.0.0.1 --port 8551
```

**Linux / macOS:**

```bash
PYTHONPATH=src python -m uvicorn TensyGridEngine.gui.dashboard:app --host 127.0.0.1 --port 8551
```

For port 8551, open **[http://localhost:8551](http://localhost:8551)**. These commands require you to choose the alternative port; they do not automatically switch ports.

## 7. Start it again later

Open a terminal in `VeraGrid_TenSyGrid`, activate the existing virtual environment as in step 3, and repeat the startup command from step 5 with an available port. You do not need to clone the repository or create the environment again.

After stopping the server, you can leave the virtual environment with:

```bash
deactivate
```

For the broader VeraGrid package and development setup, see [BUILD.md](BUILD.md).
