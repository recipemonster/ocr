import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


APP_DIRECTORY = Path(__file__).resolve().parent
MODEL_DIRECTORY = Path(os.getenv("OCR_MODEL_DIRECTORY", "/models"))
RUNTIME_DIRECTORY = MODEL_DIRECTORY / "runtime"
PACKAGE_CACHE_DIRECTORY = MODEL_DIRECTORY / "uv-cache"
DEVICE_MODE = os.getenv("OCR_DEVICE", "auto").strip().lower()
RUNTIME_SCHEMA_VERSION = "5"


def cuda_device_visible():
    return Path("/dev/nvidiactl").exists() and any(Path("/dev").glob("nvidia[0-9]*"))


def runtime_profile():
    if DEVICE_MODE not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("OCR_DEVICE must be auto, cpu or cuda")
    if DEVICE_MODE == "cuda":
        return "gpu"
    if DEVICE_MODE == "cpu":
        return "cpu"
    return "gpu" if cuda_device_visible() else "cpu"


def runtime_identity(profile):
    digest = hashlib.sha256()
    digest.update(f"runtime-schema-{RUNTIME_SCHEMA_VERSION}\n".encode())
    digest.update(f"python-{sys.version_info.major}.{sys.version_info.minor}\n".encode())
    for name in ("requirements-base.txt", f"requirements-{profile}.txt"):
        digest.update(name.encode())
        digest.update((APP_DIRECTORY / name).read_bytes())
    return digest.hexdigest()[:16]


def install_runtime(profile, destination):
    temporary = destination.with_name(f".{destination.name}.installing-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)

    environment = os.environ.copy()
    environment["UV_NO_PROGRESS"] = "1"
    environment["UV_PYTHON_DOWNLOADS"] = "never"
    subprocess.run(
        [
            "uv",
            "venv",
            "--python",
            sys.executable,
            "--no-managed-python",
            str(temporary),
        ],
        check=True,
        cwd=APP_DIRECTORY,
        env=environment,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(temporary / "bin" / "python"),
            "--cache-dir",
            str(PACKAGE_CACHE_DIRECTORY),
            "--index-strategy",
            "unsafe-best-match",
            "--requirement",
            str(APP_DIRECTORY / "requirements-base.txt"),
            "--requirement",
            str(APP_DIRECTORY / f"requirements-{profile}.txt"),
        ],
        check=True,
        cwd=APP_DIRECTORY,
        env=environment,
    )
    subprocess.run(
        [
            str(temporary / "bin" / "python"),
            str(APP_DIRECTORY / "license_inventory.py"),
            "--output",
            str(temporary / "licenses"),
        ],
        check=True,
        cwd=APP_DIRECTORY,
        env=environment,
    )
    (temporary / "runtime.json").write_text(
        json.dumps({"profile": profile, "identity": destination.name}, separators=(",", ":")),
        encoding="utf-8",
    )
    shutil.rmtree(destination, ignore_errors=True)
    temporary.rename(destination)


def main():
    profile = runtime_profile()
    identity = runtime_identity(profile)
    destination = RUNTIME_DIRECTORY / f"{profile}-{identity}"
    marker = destination / "runtime.json"
    RUNTIME_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PACKAGE_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with (RUNTIME_DIRECTORY / ".bootstrap.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not marker.is_file():
            print(f"installing OCR {profile} runtime into {destination}", flush=True)
            install_runtime(profile, destination)
        else:
            print(f"using cached OCR {profile} runtime from {destination}", flush=True)

    environment = os.environ.copy()
    environment["OCR_RUNTIME_PROFILE"] = profile
    python = destination / "bin" / "python"
    os.execve(python, [str(python), str(APP_DIRECTORY / "main.py")], environment)


if __name__ == "__main__":
    main()
