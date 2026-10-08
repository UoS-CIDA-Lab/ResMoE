"""Validate the tools required to score MultiPL-E Java programs."""
from __future__ import annotations

import shutil
from pathlib import Path


def require_java_environment(jar: Path) -> None:
    """Fail before generation when the Java evaluation environment is incomplete."""
    missing = [tool for tool in ("javac", "java") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "MultiPL-E Java requires a JDK with javac and java on PATH; "
            f"missing: {', '.join(missing)}. Follow the Java setup instructions in README.md."
        )
    if not jar.is_file():
        raise RuntimeError(
            f"MultiPL-E Java dependency is missing: {jar}. "
            "Restore the tracked mple_java/lib/javatuples-1.2.jar file; "
            "see the Java setup instructions in README.md."
        )
