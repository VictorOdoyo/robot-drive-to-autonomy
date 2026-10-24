def provenance(case, parameters):
    import hashlib
    import json
    import subprocess
    from pathlib import Path

    from . import __version__

    payload = json.dumps(
        {"case": case, "parameters": parameters}, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    root = Path(__file__).resolve().parents[2]
    revision = None
    if (root / "pyproject.toml").is_file() and (root / "src" / "robotlab" / "__init__.py").is_file():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
            )
            revision = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return {
        "package_version": __version__,
        "git_revision": revision,
        "configuration_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
