"""Build the self-contained Plotly Cloud upload directory."""

from __future__ import annotations

import shutil
from pathlib import Path


def copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == "__pycache__" or item.name.startswith(".") or item.suffix in {".pyc", ".pyo"}:
            continue
        target = destination / item.name
        if item.is_dir():
            copy_tree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    destination = (root / "deploy" / "plotly-cloud").resolve()
    expected = (root / "deploy" / "plotly-cloud").resolve()
    if destination != expected or root not in destination.parents:
        raise RuntimeError(f"refusing to replace unexpected path: {destination}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    shutil.copy2(root / "app.py", destination / "app.py")
    shutil.copy2(root / "scripts" / "cloud_main.py", destination / "main.py")
    shutil.copy2(root / "requirements-cloud.txt", destination / "requirements.txt")
    shutil.copy2(root / "docs" / "PLOTLY_CLOUD_DEPLOYMENT.md", destination / "README.md")
    copy_tree(root / "packetflowai", destination / "packetflowai")

    print(f"Plotly Cloud package prepared: {destination}")


if __name__ == "__main__":
    main()
