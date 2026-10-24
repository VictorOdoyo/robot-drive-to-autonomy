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


def save_archive(result, path):
    import json
    import os
    import tempfile
    from pathlib import Path

    validate_result(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in result.items() if key not in arrays}
    metadata["schema_version"] = 1
    metadata["provenance"] = result.get("provenance") or provenance(result["case"], result["parameters"])
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = stream.name
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()
    return path


def load_archive(path):
    import json

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata.pop("schema_version", None) != 1:
            raise ValueError("unsupported mission archive schema")
        result = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
    if set(result) & set(metadata):
        raise ValueError("archive metadata shadows numeric arrays")
    result.update(metadata)
    return validate_result(result)


def export_csv(result, path):
    from pathlib import Path

    validate_result(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = result["state"]
    matrix = np.column_stack(
        [
            result["time"],
            state[:, :14],
            result["estimate"],
            result["command"],
            result["duty"],
            result["bus_voltage"],
            result["battery_current"],
            result["clearance"],
            state[:, 16:18],
        ]
    )
    names = [
        "time_s",
        "left_current_a",
        "right_current_a",
        "left_wheel_rad_s",
        "right_wheel_rad_s",
        "forward_m_s",
        "yaw_rad_s",
        "x_m",
        "y_m",
        "heading_rad",
        "soc",
        "polarization_v",
        "left_motor_c",
        "right_motor_c",
        "battery_c",
        "estimated_x_m",
        "estimated_y_m",
        "estimated_heading_rad",
        "estimated_gyro_bias_rad_s",
        "command_m_s",
        "command_rad_s",
        "left_duty",
        "right_duty",
        "bus_v",
        "battery_current_a",
        "clearance_m",
        "dissipated_j",
        "external_work_j",
    ]
    np.savetxt(path, matrix, delimiter=",", header=",".join(names), comments="")
    return path


def plot_mission(result, path):
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    from .analysis import mission_metrics
    from .electrical import Motor
    from .mechanics import Chassis
    from .plant import PlantParameters, stored_energy
    from .power import Battery

    validate_result(result)
    parameters = dict(result["parameters"])
    parameters["motor"] = Motor(**parameters["motor"])
    parameters["chassis"] = Chassis(**parameters["chassis"])
    parameters["battery"] = Battery(**parameters["battery"])
    plant = PlantParameters(**parameters)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time, state = result["time"], result["state"]
    metrics = mission_metrics(result)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), layout="constrained")
    ax = axes[0, 0]
    for rectangle in result["case"]["shelves"]:
        ax.add_patch(
            Rectangle(
                rectangle[:2],
                rectangle[2] - rectangle[0],
                rectangle[3] - rectangle[1],
                facecolor="#334155",
                alpha=0.8,
            )
        )
    ax.plot(*result["route"].T, "--", color="#94a3b8", label="planned route")
    ax.plot(state[:, 6], state[:, 7], color="#2563eb", label="true path")
    ax.plot(result["estimate"][:, 0], result["estimate"][:, 1], color="#d97706", alpha=0.6, label="EKF")
    for index in np.linspace(0, len(time) - 1, 12, dtype=int):
        ax.add_patch(
            Circle(state[index, 6:8], result["case"]["radius"], fill=False, color="#2563eb", alpha=0.25)
        )
    ax.scatter(*result["case"]["goal"], marker="*", s=100, color="#16a34a", label="goal")
    ax.set(
        xlim=(0, result["case"]["width"]),
        ylim=(0, result["case"]["height"]),
        xlabel="x (m)",
        ylabel="y (m)",
        title="Circular footprint and warehouse route",
        aspect="equal",
    )
    ax.legend(fontsize=8)
    axes[0, 1].plot(time, state[:, 4], label="true speed")
    axes[0, 1].plot(time, result["command"][:, 0], "--", label="command")
    axes[0, 1].set(title="Forward motion", ylabel="Speed (m/s)")
    for k, label in enumerate(("left", "right")):
        axes[1, 0].plot(time, state[:, k], color=("#2563eb", "#d97706")[k], label=label)
        axes[1, 0].plot(
            time, result["current_reference"][:, k], "--", color=("#2563eb", "#d97706")[k], alpha=0.45
        )
        axes[2, 0].plot(time, state[:, 11 + k], label=label)
    axes[1, 0].set(title="Winding current; dashed = reference", ylabel="Current (A)")
    axes[1, 1].plot(time, result["bus_voltage"], color="#7c3aed")
    axes[1, 1].set(title="Loaded battery terminal voltage", ylabel="Voltage (V)")
    axes[2, 0].set(title="Motor winding temperatures", ylabel="Temperature (°C)")
    stored = np.array([stored_energy(plant, row) for row in state])
    axes[2, 1].plot(time, stored[0] - stored, label="stored energy used")
    axes[2, 1].plot(
        time, state[:, 16] - state[0, 16] + state[:, 17] - state[0, 17], "--", label="loss + external work"
    )
    axes[2, 1].set(title="Electrochemical / mechanical energy ledger", ylabel="Energy (J)")
    error = np.linalg.norm(state[:, 6:8] - result["estimate"][:, :2], axis=1)
    sigma = np.sqrt(np.linalg.eigvalsh(result["covariance"][:, :2, :2])[:, -1])
    axes[3, 0].plot(time, error, label="position error")
    axes[3, 0].plot(time, 3 * sigma, "--", label="3 x largest position sigma")
    axes[3, 0].set(title="Localization error and covariance scale", ylabel="Position (m)")
    axes[3, 1].plot(time, result["clearance"], color="#059669")
    axes[3, 1].axhline(0, color="#dc2626", linestyle="--")
    axes[3, 1].set(title="True footprint clearance", ylabel="Clearance (m)")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        if ax is not axes[0, 0]:
            ax.set_xlabel("Time (s)")
        if ax.get_legend_handles_labels()[0] and ax is not axes[0, 0]:
            ax.legend(fontsize=8)
    axes[2, 0].ticklabel_format(axis="y", useOffset=False, style="plain")
    fig.suptitle(
        f"{result['case']['name']} | simulated robot mission\n"
        f"Goal error {metrics['goal_error_m']:.3f} m | minimum clearance {metrics['minimum_clearance_m']:.3f} m\n"
        f"Energy ledger residual {metrics['energy_residual_j']:.2e} J",
        fontsize=13,
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_report(result, directory):
    import hashlib
    import html
    import json
    from pathlib import Path

    from .analysis import consistency, mission_metrics

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_archive(result, directory / "mission.npz")
    export_csv(result, directory / "telemetry.csv")
    plot_mission(result, directory / "mission.png")
    metrics = mission_metrics(result)
    summary = consistency(result)
    metrics["nees_mean"] = summary["mean_nees"]
    metrics["nees_95_coverage"] = summary["coverage"]
    (directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    source = result.get("provenance") or provenance(result["case"], result["parameters"])
    (directory / "provenance.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = "".join(
        "<tr><th>" + html.escape(key.replace("_", " ")) + "</th><td>" + html.escape(str(value)) + "</td></tr>"
        for key, value in metrics.items()
    )
    title = html.escape(result["case"]["name"])
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} | Robotlab</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;color:#172033}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;padding:.5rem;border-bottom:1px solid #dce1ea}}
th{{font-weight:500}}a{{color:#164ac4}}</style>
<h1>{title}</h1><p>Simulated differential-drive robot. SI-unit electrical, mechanical and estimation traces.
These results describe the specified mathematical models; no hardware validation is claimed.</p>
<p><a href="mission.npz">Replay archive</a> | <a href="telemetry.csv">SI telemetry</a> |
<a href="metrics.json">Metrics</a> | <a href="provenance.json">Source and configuration</a></p>
<img src="mission.png" alt="Route, winding current, voltage, temperature, energy and localization traces">
<h2>Measured results</h2><table>{rows}</table>
<p>Configuration SHA-256: <code>{html.escape(source["configuration_sha256"])}</code></p></html>"""
    (directory / "index.html").write_text(document, encoding="utf-8")
    filenames = [
        "mission.npz",
        "telemetry.csv",
        "mission.png",
        "metrics.json",
        "provenance.json",
        "index.html",
    ]
    manifest = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in filenames}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory / "index.html"


def verify_report(directory):
    import hashlib
    import json
    from pathlib import Path

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("empty or invalid report manifest")
    for name, expected in manifest.items():
        if Path(name).name != name:
            raise ValueError("manifest entries must be local filenames")
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("report integrity mismatch: " + name)
    return True
