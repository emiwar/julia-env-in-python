# Multithreaded MuJoCo env with Python/Julia interop

## Installation
```
python -m venv .env
source ./.env/bin/activate
pip install -r requirements.txt
```
This will install a local Julia environment in .env/julia_env, and install MuJoCo.jl there. The version is specified in `juliapkg.json`.

## Running
```
source ./.env/bin/activate
export PYTHON_JULIACALL_THREADS=auto
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
python main.py
```
This sets the number of threads to be automatically determined by Julia, which is usually the number of CPU cores. Alternatively, `auto` can be replaced with an integer to fix the number of threads.