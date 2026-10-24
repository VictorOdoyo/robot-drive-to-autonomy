"""Generate published evidence from validated archives; never invent measurements."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from robotlab.analysis import mission_metrics
from robotlab.cases import experiment_catalog, from_dict
from robotlab.plant import PlantParameters
from robotlab.reporting import load_archive, plot_mission, save_archive, verify_report
from robotlab.studies import copper_thermal_analysis, current_stability, drive_lqr, mapping_reconstruction


def control_figure(output):
    periods = np.geomspace(0.0001, 0.005, 48)
    bandwidths = np.geomspace(30, 4000, 100)
    direct = current_stability(periods, bandwidths)
    delayed = current_stability(periods, bandwidths, delayed=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, values, title in zip(
        axes, (direct, delayed), ("No transport delay", "One-cycle transport delay"), strict=True
    ):
        mesh = ax.pcolormesh(
            bandwidths,
            periods * 1000,
            np.log10(values),
            shading="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5),
        )
        ax.contour(bandwidths, periods * 1000, values, levels=[1], colors="black", linewidths=1)
        ax.scatter([150], [2], s=65, color="white", edgecolor="black", label="mission setting")
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="PI bandwidth (rad/s)",
            ylabel="Sample period (ms)",
            title=title,
        )
        ax.legend(fontsize=8)
    fig.colorbar(mesh, ax=axes, label="log10 spectral radius; below zero is Schur stable")
    fig.suptitle("Sampled winding-current PI stability | exact held-speed electrical plant")
    fig.savefig(output / "control-stability.png", dpi=140)
    plt.close(fig)
    np.savez_compressed(
        output / "control-stability.npz",
        period_s=periods,
        bandwidth_rad_s=bandwidths,
        direct_radius=direct,
        delayed_radius=delayed,
    )
    design = drive_lqr(PlantParameters())
    summary = {
        "direct_stable_fraction": float(np.mean(direct < 1)),
        "delayed_stable_fraction": float(np.mean(delayed < 1)),
        "lqr_poles": [{"real": float(p.real), "imaginary": float(p.imag)} for p in design["poles"]],
        "copper_thermal_5a": copper_thermal_analysis(5),
        "notes": "Stability fractions describe this sampling grid, not probabilities.",
    }
    (output / "control-study.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def map_figure(result, output):
    learned = mapping_reconstruction(result, scan_stride=20)
    case = result["case"]
    extent = (0, case["width"], 0, case["height"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, probability, title in zip(
        axes,
        (result["map_probability"], learned.probability()),
        ("Known map used for planning", "Offline simulated lidar evidence"),
        strict=True,
    ):
        image = ax.imshow(probability, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
        ax.plot(result["state"][:, 6], result["state"][:, 7], color="#2563eb", linewidth=1)
        ax.set(xlabel="x (m)", ylabel="y (m)", title=title, aspect="equal")
    fig.colorbar(image, ax=axes, label="Occupancy probability; 0.5 means unknown")
    fig.suptitle("Warehouse mapping study | noisy EKF poses and range measurements")
    fig.savefig(output / "mapping.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/catalog"))
    parser.add_argument("--output", type=Path, default=Path("docs/gallery"))
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for index, (name, case) in enumerate(experiment_catalog().items()):
        folder = arguments.input / f"{index + 1:02d}"
        verify_report(folder)
        result = load_archive(folder / "mission.npz")
        if from_dict(result["case"]) != case:
            raise ValueError(f"{name}: archive configuration differs from current catalog; rerun it")
        evidence[name] = {
            "metrics": mission_metrics(result),
            "provenance": result["provenance"],
            "case": result["case"],
        }
        if name in ("warehouse", "emergency-stop"):
            plot_mission(result, arguments.output / f"{name}.png")
            save_archive(result, arguments.output / f"{name}.npz")
        if name == "warehouse":
            map_figure(result, arguments.output)
    (arguments.output / "experiments.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = [
        "| Experiment | Goal reached | Goal error (m) | Min clearance (m) | Peak current (A) | Position RMSE (m) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, entry in evidence.items():
        values = entry["metrics"]
        rows.append(
            f"| {name} | {values['completed']} | {values['goal_error_m']:.3f} | "
            f"{values['minimum_clearance_m']:.3f} | {values['peak_current_a']:.3f} | "
            f"{values['position_rmse_m']:.3f} |"
        )
    (arguments.output / "results.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    control_figure(arguments.output)
    print(json.dumps({"experiments": len(evidence), "output": str(arguments.output.resolve())}))


if __name__ == "__main__":
    main()
