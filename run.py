"""Start the whole web app with one command:  python run.py   (options: --port 8000, --no-browser)

Installs missing Python packages, installs/builds the frontend when needed, starts the
FastAPI server (which also serves the built frontend) and opens the browser."""
import argparse, importlib.util, os, shutil, subprocess, sys, threading, time, urllib.request, webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONT = os.path.join(ROOT, "frontend")
DIST_INDEX = os.path.join(FRONT, "dist", "index.html")


def step(msg): print(f"\n==> {msg}", flush=True)


def run(cmd, cwd=ROOT):
    subprocess.run(cmd, cwd=cwd, check=True)


def newest_mtime(*paths):
    t = 0.0
    for p in paths:
        if os.path.isfile(p):
            t = max(t, os.path.getmtime(p))
        for d, dirs, files in os.walk(p):
            dirs[:] = [x for x in dirs if x not in ("node_modules", "dist")]
            t = max([t] + [os.path.getmtime(os.path.join(d, f)) for f in files])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    if not os.path.isfile(os.path.join(ROOT, "edsr_x4.pt")):
        sys.exit("edsr_x4.pt not found in the repo root. Copy the trained weights there first.")
    if importlib.util.find_spec("torch") is None:
        sys.exit("PyTorch is not installed. Install it first: https://pytorch.org/get-started/locally/")

    missing = [m for m in ("fastapi", "uvicorn", "multipart", "skimage", "PIL", "numpy") if importlib.util.find_spec(m) is None]
    if missing:
        step(f"Installing Python packages (missing: {', '.join(missing)})")
        run([sys.executable, "-m", "pip", "install", "-r", os.path.join("backend", "requirements.txt")])

    npm = shutil.which("npm")
    if not npm:
        sys.exit("npm not found. Install Node.js 20+ from https://nodejs.org")
    if not os.path.isdir(os.path.join(FRONT, "node_modules")):
        step("Installing frontend packages (first run only)")
        run([npm, "install"], cwd=FRONT)
    src = [os.path.join(FRONT, p) for p in ("src", "index.html", "package.json", "vite.config.ts")]
    if not os.path.isfile(DIST_INDEX) or newest_mtime(*src) > os.path.getmtime(DIST_INDEX):
        step("Building frontend")
        run([npm, "run", "build"], cwd=FRONT)

    url = f"http://localhost:{a.port}"

    def open_when_ready():
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{url}/api/health", timeout=1)
                print(f"\n==> Ready: {url}   (Ctrl+C to stop)\n", flush=True)
                if not a.no_browser:
                    webbrowser.open(url)
                return
            except OSError:
                time.sleep(0.5)
        print("\n!! Server did not become ready; check the log above.", flush=True)

    step(f"Starting server on {url} (loading model...)")
    threading.Thread(target=open_when_ready, daemon=True).start()
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "backend.app:app", "--port", str(a.port)], cwd=ROOT)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
