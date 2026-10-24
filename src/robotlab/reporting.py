import numpy as np


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


def validate_result(result):
    from .cases import from_dict

    from_dict(result["case"])
    time = np.asarray(result["time"])
    if time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("strictly increasing finite trace time required")
    shapes = {
        "state": (18,),
        "estimate": (4,),
        "covariance": (4, 4),
        "command": (2,),
        "duty": (2,),
        "current_reference": (2,),
        "wheel_reference": (2,),
        "bus_voltage": (),
        "battery_current": (),
        "slip_speed": (2,),
        "clearance": (),
        "fault": (),
    }
    for name, tail in shapes.items():
        array = np.asarray(result[name])
        if array.shape != (len(time), *tail) or array.dtype.kind not in "biufU":
            raise ValueError("invalid trace array: " + name)
        if array.dtype.kind in "iuf" and not np.all(np.isfinite(array)):
            raise ValueError("nonfinite trace array: " + name)
    if not np.allclose(result["covariance"], np.swapaxes(result["covariance"], 1, 2), atol=1e-10):
        raise ValueError("asymmetric covariance trace")
    if np.min(np.linalg.eigvalsh(result["covariance"])) < -1e-10:
        raise ValueError("indefinite covariance trace")
    if np.any(np.abs(result["duty"]) > 1 + 1e-10):
        raise ValueError("bridge duty outside physical bounds")
    return result
