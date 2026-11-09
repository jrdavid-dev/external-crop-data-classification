"""First-run notebook setup using only Python's standard library.

No setup or downloads happen on import. Source notebooks explicitly call these
helpers before importing scientific packages or retrieving reference files.
"""

import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_SETTINGS = {
    "dataset_root": "./ground_truth",
    "imagery_root": "./ground_truth/sentinel2_pilot_composites",
    "report_root": "./sen4agrinet_work/inventory",
    "annual_archive_root": "./sen4agrinet_work/annual_archive",
    "analysis_collection": "adaptive_training",
    "earth_engine_project": "",
}


def configure_certificates():
    """Configure trust before the first pip, Hugging Face or label request."""
    try:
        import certifi
    except ImportError:
        return
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())


def required_packages(requirements_path):
    return [
        line.strip()
        for line in Path(requirements_path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def missing_packages(requirements):
    """Keep working installed packages; install missing distributions in this kernel."""
    missing = []
    for requirement in requirements:
        name = re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].strip()
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(requirement)
    return missing


def prepare_environment(repository_root, install_missing=True):
    """Make a cloned src layout importable and install missing notebook dependencies."""
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "Choose a Python 3.12+ notebook kernel; Python 3.13 is recommended."
        )
    repository_root = Path(repository_root).resolve()
    source_folder = str(repository_root / "src")
    if source_folder not in sys.path:
        sys.path.insert(0, source_folder)
    # Use the package checkout metadata when the kernel starts from another working folder.
    requirements_path = repository_root / "requirements-notebooks.txt"
    if not requirements_path.is_file():
        requirements_path = (
            Path(__file__).resolve().parents[2] / "requirements-notebooks.txt"
        )
    configure_certificates()
    missing = missing_packages(required_packages(requirements_path))
    if missing and not install_missing:
        raise RuntimeError("Offline setup requires installed dependencies: " + ", ".join(missing))
    if missing:
        print("Installing missing packages in the active kernel:", missing)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        importlib.invalidate_caches()
    configure_certificates()
    package_root = (
        repository_root
        if (repository_root / "pyproject.toml").is_file()
        else requirements_path.parent
    )
    if install_missing:
        ensure_editable_package(package_root)
    print("Notebook environment ready:", sys.executable)


def editable_package_root():
    """Fresh downstream kernels must import the same clone, rather than another checkout."""
    from urllib.parse import urlparse

    try:
        distribution = importlib.metadata.distribution("crop-pipeline")
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if not direct_url.get("dir_info", {}).get("editable"):
        return None
    parsed = urlparse(direct_url.get("url", ""))
    if parsed.scheme != "file":
        return None
    from urllib.request import url2pathname

    return Path(url2pathname(parsed.path)).resolve()


def ensure_editable_package(repository_root):
    repository_root = Path(repository_root).resolve()
    if editable_package_root() == repository_root:
        return
    print("Installing the cloned pipeline for fresh notebook kernels:", repository_root)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-e",
            str(repository_root),
        ]
    )
    importlib.invalidate_caches()


def read_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError(f"Expected a JSON settings object in {path}")
    return settings


def ensure_local_settings(path):
    """Create portable defaults once; never replace someone's existing local paths."""
    path = Path(path)
    if path.exists():
        return read_settings(path)
    settings = {
        **DEFAULT_SETTINGS,
        "earth_engine_project": os.environ.get("EE_PROJECT", "").strip(),
    }
    return publish_initial_settings(path, settings)


def publish_initial_settings(path, settings):
    """Another kernel may read the new file immediately, so publish complete JSON atomically."""
    import tempfile

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".settings_", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(settings, stream, indent=2)
            stream.write("\n")
        return link_initial_settings(temporary, path, settings)
    finally:
        temporary.unlink(missing_ok=True)


def link_initial_settings(temporary, path, settings):
    try:
        os.link(temporary, path)
    except FileExistsError:
        return read_settings(path)
    print("Created local settings:", path.resolve())
    return settings


def project_for_collection(settings, settings_path, required):
    """Ask once for account-specific setup; project IDs cannot be inferred from a clone."""
    project = (
        os.environ.get("EE_PROJECT", "").strip()
        or str(settings.get("earth_engine_project", "")).strip()
    )
    if project and project != "YOUR_EARTH_ENGINE_PROJECT":
        return project
    if not required:
        return ""
    project = input(
        "Enter your registered Earth Engine Google Cloud project ID (saved locally): "
    ).strip()
    if (
        not project
        or project == "YOUR_EARTH_ENGINE_PROJECT"
        or any(character.isspace() for character in project)
    ):
        raise ValueError(
            "A registered Earth Engine project ID is required before collection. Rerun this cell with your project ID."
        )
    latest_settings = (
        read_settings(settings_path) if Path(settings_path).exists() else dict(settings)
    )
    latest_settings["earth_engine_project"] = project
    save_settings(settings_path, latest_settings)
    settings.update(latest_settings)
    return project


def save_settings(path, settings):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
