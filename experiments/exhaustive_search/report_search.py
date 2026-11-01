"""Render the completed finite search audit without choosing or changing models.

This script refuses to write a report until every required assessment and every
planned screen/nested record exists. Failure records count as completed attempts
and are reported explicitly. The promotion gate is read from frozen assessments;
it does not introduce another candidate or selection rule.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
LIBOMP_DIRECTORY = "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/sklearn/.dylibs"
REQUIRED = (
    "protocol.json", "stability_protocol.json", "selection_screen.json",
    "screen_family_summary.json", "nested_assessment.json", "repeated_results.json",
)


def load_completed(directory=HERE):
    directory = Path(directory)
    missing = [name for name in REQUIRED if not (directory / name).is_file()]
    if missing:
        raise RuntimeError("Report not written: required completed artifacts are missing: "
                           + ", ".join(missing))
    artifacts = {name.removesuffix(".json"): json.loads((directory / name).read_text())
                 for name in REQUIRED}
    plan = artifacts["protocol"]
    for relative, expected in plan["sha256"].items():
        path = directory.parents[1] / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source or data changed: {relative}")
    expected_specs = {spec["id"]: spec for spec in plan["specs"]}
    phases = {}
    for phase, definition in plan["phases"].items():
        records = []
        for identifier in definition["ids"]:
            path = directory / "results" / phase / f"{identifier}.json"
            if not path.is_file():
                raise RuntimeError(f"Report not written: incomplete phase {phase}, missing {identifier}.")
            record = json.loads(path.read_text())
            if record.get("id") != identifier or record.get("spec") != expected_specs[identifier]:
                raise ValueError(f"Record differs from frozen specification: {phase}/{identifier}")
            if "error" not in record:
                if len(record.get("folds", [])) != len(definition["splits"]):
                    raise ValueError(f"Missing fold metrics: {phase}/{identifier}")
                if not path.with_suffix(".npz").is_file():
                    raise ValueError(f"Missing prediction artifact: {phase}/{identifier}")
            records.append(record)
        phases[phase] = records
    expected_hash = hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest()
    screen = artifacts["selection_screen"]
    if screen["protocol_sha256"] != expected_hash:
        raise ValueError("Screen selection uses a different protocol.")
    repeated = artifacts["repeated_results"]
    if repeated["provenance"]["protocol.json"] != expected_hash:
        raise ValueError("Repeated validation uses a different protocol.")
    if repeated["provenance"]["selection_screen.json"] != hashlib.sha256(
            (directory / "selection_screen.json").read_bytes()).hexdigest():
        raise ValueError("Repeated validation uses a different screen-selected recipe.")
    if len(repeated["folds"]) != len(artifacts["stability_protocol"]["folds"]):
        raise ValueError("Repeated assessment does not cover every planned fold.")
    nested = artifacts["nested_assessment"]
    if nested["outer_folds"] != len(plan["outer_splits"]) or len(nested["folds"]) != len(plan["outer_splits"]):
        raise ValueError("Nested assessment does not cover every planned outer fold.")
    for index in range(len(plan["outer_splits"])):
        path = directory / f"selection_nested_{index}.json"
        if not path.is_file() or json.loads(path.read_text())["protocol_sha256"] != expected_hash:
            raise ValueError(f"Nested selection {index} is missing or uses another protocol.")
    return artifacts, phases


def promotion_gate(artifacts):
    repeated = artifacts["repeated_results"]["summary"]["compatible"]
    nested = artifacts["nested_assessment"]["procedures"]["compatible"]
    repeated_passes = bool(repeated.get("complete") and
                           repeated.get("passes_repeated_portion_of_promotion_rule"))
    nested_passes = bool("error" not in nested and all(
        nested.get("mean_delta_from_baseline", {}).get(metric, -np.inf) > 0
        for metric in ("auc", "tpr")))
    return {"repeated_passes": repeated_passes, "nested_passes": nested_passes,
            "eligible_for_promotion": repeated_passes and nested_passes}


def metric_text(metrics):
    return f"AUC {metrics['auc']:.12f}; TPR at 1% FPR {100 * metrics['tpr']:.9f}%"


def delta_text(delta):
    return f"AUC {delta['auc']:+.9f}; TPR {100 * delta['tpr']:+.6f} percentage points"


def recipe_text(recipe):
    parts = []
    for spec, weight, identifier in zip(recipe["specs"], recipe["weights"], recipe["model_ids"]):
        params = dict(spec["params"])
        family = params.pop("family", spec["source"])
        details = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
        parts.append(f"{weight:.6g} × {family} ({details}; ID `{identifier}`)" if details else
                     f"{weight:.6g} × {family} (ID `{identifier}`)")
    return recipe["method"] + " combination: " + "; ".join(parts) + "."


def make_plot(screen_records, selections, destination):
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spam-search-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [record for record in screen_records if "error" not in record]
    current = selections["compatible"]["baseline"]
    x = np.array([record["auc"] for record in valid])
    y = 100 * np.array([record["tpr"] for record in valid])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.7), sharey=False)
    for axis in axes:
        axis.scatter(x, y, c="#a7abb1", s=10, alpha=.45, linewidths=0,
                     label=f"Successful configurations ({len(valid):,})")
        axis.scatter([current["auc"]], [100 * current["tpr"]], c="black", marker="x",
                     s=95, linewidths=2.2, label="Current submission", zorder=5)
        for procedure, color, marker, label in (
                ("compatible", "#2166ac", "*", "Selected compatible recipe"),
                ("all_libraries", "#b56a21", "D", "Selected all-library recipe")):
            choice = selections[procedure]["chosen"]
            axis.scatter([choice["auc"]], [100 * choice["tpr"]], edgecolors=color,
                         facecolors="none", marker=marker, s=155 if marker == "*" else 85,
                         linewidths=1.5, label=label, zorder=6)
        axis.set_xlabel("Mean fold ROC AUC (unitless)")
        axis.set_ylabel("Mean TPR at FPR = 1% (% of spam)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#e4e5e7", linewidth=.6, zorder=0)
        axis.set_axisbelow(True)
    axes[0].set_title("Entire finite configuration library", fontsize=11)
    axes[1].set_title("Near the current submission", fontsize=11)
    chosen_auc = [selections[name]["chosen"]["auc"] for name in selections]
    chosen_tpr = [100 * selections[name]["chosen"]["tpr"] for name in selections]
    axes[1].set_xlim(current["auc"] - .025, max(float(x.max()), *chosen_auc) + .003)
    axes[1].set_ylim(max(0, 100 * current["tpr"] - 15), max(float(y.max()), *chosen_tpr) + 4)
    fig.suptitle("Development tradeoffs: ranking quality and low-false-positive detection", fontsize=13, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .075), ncol=2,
               frameon=False, fontsize=9)
    fig.text(.5, .022,
             "Source: spamTrain1.csv + spamTrain2.csv, 3,000 released emails; fixed three-fold screen.\n"
             "Points are fold means. Recipes were selected on these same folds. Development evidence only; final test remains hidden.",
             ha="center", va="bottom", fontsize=8.5)
    fig.subplots_adjust(left=.075, right=.985, top=.86, bottom=.25, wspace=.23)
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)


def render_readme(artifacts, phases, include_plot=True):
    plan = artifacts["protocol"]
    screen = artifacts["selection_screen"]
    selections = screen["procedures"]
    nested = artifacts["nested_assessment"]
    repeated = artifacts["repeated_results"]
    stability = artifacts["stability_protocol"]
    gate = promotion_gate(artifacts)
    family_counts = Counter(f"{spec['source']}:{spec['params']['family']}" for spec in plan["specs"])
    source_counts = Counter(spec["source"] for spec in plan["specs"])
    planned_slots = sum(len(phase["ids"]) * len(phase["splits"]) for phase in plan["phases"].values())
    successful_slots = sum(len(record.get("folds", [])) for rows in phases.values() for record in rows)
    outer_refits = sum(len(fold["fitted_members"]) for fold in nested["folds"])
    repeated_refits = sum(len(fold["members"]) for fold in repeated["folds"])
    primary = selections["compatible"]
    baseline_screen = primary["baseline"]
    dominating = [record for record in phases["screen"] if "error" not in record
                  and record["spec"]["source"] != "current"
                  and record["auc"] > baseline_screen["auc"] + 1e-12
                  and record["tpr"] > baseline_screen["tpr"] + 1e-12]
    verdict = ("The compatible screen-selected recipe passes the prespecified promotion gate."
               if gate["eligible_for_promotion"] else
               "Retain the current submission: the compatible screen-selected recipe fails the prespecified promotion gate.")
    lines = [
        "# Broad spam-classifier search",
        "", verdict, "",
        "This report records the analysis gate. A passing candidate still requires an exact "
        "submission implementation and separate interface/parity verification; this script does not change `classifySpam.py`.",
        "", "## Scope and frozen procedure", "",
        f"- **{len(plan['specs']):,} unique constructor-expanded configurations**, from "
        f"{len(plan['specs']) + len(plan['aliases']):,} serialized specifications; "
        f"{len(plan['aliases'])} default aliases were deduplicated. The menu has {len(family_counts)} "
        "source/family labels; preprocessing variants are not separate fundamental algorithms.",
        f"- All **3,000 released emails** are development data. The screen uses three folds. "
        f"Nested validation repeats the full {len(plan['nested_ids']):,}-configuration search inside "
        f"each of {len(plan['outer_splits'])} outer training partitions, with three inner folds per partition.",
        f"- The four search phases contain **{planned_slots:,} planned configuration/fold fit slots**; "
        f"{successful_slots:,} successful fold results were recorded. There were also {outer_refits:,} "
        f"selected-member outer refit attempts and {repeated_refits:,} member refit attempts across "
        f"{len(stability['folds'])} repeated-check folds. The current three-component ensemble counts "
        "as one configuration per fit here; this is not a count of every internal estimator fit.",
        "- All learned imputation, scaling, binning, feature selection, and model fitting use the "
        "training partition only. Raw decision margins are retained for ranking-only classifiers.",
        "- Selection requires mean fold AUC at least the current submission's, then maximizes "
        "`mean AUC + 0.05 × mean TPR`. This is a research heuristic, not the instructor's grade formula.",
        "- Probability-model pools have at most 24 members and two per family. The finite blend "
        "menu covers all declared pair weights, all declared triple weight permutations, and "
        "5,000 seeded sparse Dirichlet draws, each in probability and logit space. All singletons "
        "are also considered. Stacking was not evaluated.",
    ]
    for name, label in (("compatible", "Compatible primary search"), ("all_libraries", "All-library benchmark")):
        counts = selections[name]["counts"]
        nested_evaluations = sum(fold["procedures"][name]["selection_counts"]["total_recipe_evaluations"]
                                 for fold in nested["folds"])
        lines.append(f"- {label}: **{counts['total_recipe_evaluations']:,} screen recipe evaluations**, "
                     f"including {counts['singleton_models']:,} singletons and {counts['blend_evaluations']:,} "
                     f"blends; the nested selectors add {nested_evaluations:,} "
                     "recipe evaluations. Menus overlap across procedures and folds; these totals are not distinct mathematical algorithms.")
    expansion = plan.get("coverage_expansion")
    if expansion:
        lines += ["", f"The initial {len(plan.get('nested_pilot_ids', []))}-configuration nested pilot was "
                  "expanded to the complete library after a runtime-only check, before inspecting any "
                  "performance metrics. Data, model specifications, split IDs, selector, and timeout "
                  "limits stayed fixed. The pilot and expansion provenance are retained in "
                  "[protocol_pilot.json](protocol_pilot.json) and [protocol.json](protocol.json)."]

    lines += ["", "## Results and promotion gate", "",
              f"Screen baseline: **{metric_text(baseline_screen)}**.",
              f"The screen had **{len(dominating)}** other successful configurations that exceeded "
              "the baseline on both mean metrics. These are exploratory comparisons on the same selection folds.", ""]
    for name, label in (("compatible", "Compatible selected recipe"), ("all_libraries", "All-library selected recipe")):
        choice = selections[name]["chosen"]
        lines += [f"- {label}: **{metric_text(choice)}**.", f"- {recipe_text(choice)}"]
    lines += ["", "Nested assessment evaluates the selection procedure, with a separately selected "
              "recipe in each outer fold. It does not evaluate one fixed recipe across all outer folds."]
    if "baseline_mean_fold_metrics" in nested:
        lines += ["", f"- Nested baseline: **{metric_text(nested['baseline_mean_fold_metrics'])}**."]
    for name, label in (("compatible", "Compatible selector"), ("all_libraries", "All-library selector")):
        item = nested["procedures"][name]
        if "error" in item:
            lines.append(f"- {label}: incomplete metric assessment because {item['error']}.")
        else:
            lines.append(f"- {label}: **{metric_text(item['mean_fold_metrics'])}**; "
                         f"change from baseline: {delta_text(item['mean_delta_from_baseline'])}.")
    lines += ["", f"The repeated check holds the screen-selected recipes fixed across "
              f"{len(stability['seeds'])} repetitions of five folds, seeds "
              + ", ".join(map(str, stability["seeds"])) + "."]
    for name, label in (("baseline", "Repeated-check baseline"), ("compatible", "Compatible fixed recipe"),
                        ("all_libraries", "All-library fixed recipe")):
        item = repeated["summary"][name]
        if not item["complete"]:
            lines.append(f"- {label}: failed folds {item['failed_fold_indices']}; no successful promotion claim.")
        else:
            text = f"- {label}: **{metric_text(item['mean_fold_metrics'])}**."
            if name != "baseline":
                delta = {metric: item["paired_comparison"][metric]["mean_delta"] for metric in ("auc", "tpr")}
                text += f" Change: {delta_text(delta)}; both metrics improved in "
                text += f"{item['repetition_wins_both_metrics']}/{len(stability['seeds'])} repetition means."
            lines.append(text)
    lines += ["", "The promotion rule requires **both** mean metrics to improve in the repeated "
              "check, both to improve in at least four of five repetition means, and the compatible "
              "nested selector to improve both mean outer metrics.", "",
              f"- Repeated portion: **{'PASS' if gate['repeated_passes'] else 'FAIL'}**.",
              f"- Compatible nested portion: **{'PASS' if gate['nested_passes'] else 'FAIL'}**.",
              f"- Combined analysis gate: **{'PASS' if gate['eligible_for_promotion'] else 'FAIL'}**."]
    if include_plot:
        lines += ["", "![Screen tradeoffs, development evidence only](screen_tradeoffs.png)"]

    lines += ["", "## Coverage, failures, and warnings", ""]
    for source in sorted(source_counts):
        details = [f"{family.split(':', 1)[1]} ({count})" for family, count in sorted(family_counts.items())
                   if family.startswith(source + ":")]
        lines.append(f"- {source}: **{source_counts[source]:,} configurations** — " + ", ".join(details) + ".")
    lines += ["", "Archived `baseline` means Gaussian naive Bayes; archived `original` is an older "
              "ensemble. `current:current_submission` is the actual leaderboard-winning submission.", ""]
    for phase, rows in phases.items():
        failures = [row for row in rows if "error" in row]
        warnings = [row for row in rows if row.get("warnings")]
        lines.append(f"- {phase}: {len(rows) - len(failures):,}/{len(rows):,} configurations completed "
                     f"all folds; {len(failures)} failed; {len(warnings)} recorded non-FutureWarning notices.")
        if failures:
            reasons = Counter(row["error"] for row in failures)
            for reason, count in reasons.most_common(4):
                lines.append(f"  - {count} × {reason[:240].replace(chr(10), ' ')}")
    common = Counter(w for row in phases["screen"] for w in set(row.get("warnings", [])))
    if common:
        lines += ["", "Most frequent screen warnings (number of configuration records):"]
        lines += [f"- {count}: {notice[:260].replace(chr(10), ' ')}" for notice, count in common.most_common(5)]
    lines += ["", f"The limits were {plan['max_fit_seconds']} seconds per estimator fit and "
              f"{plan['max_preprocessing_seconds']} seconds per shared preprocessing operation. "
              "Warnings are retained rather than automatically treated as failed fits. Full failure "
              "messages and warnings are in the per-phase records and selection audit objects.",
              "", "## Reproduction and environment", "",
              "Use the saved protocol and task-local virtual environment. Source and data hashes "
              "are checked before fitting. Existing complete caches are resumed; do not reuse them "
              "after changing data, code, or specifications. `search.py init` intentionally refuses "
              "to overwrite the saved expanded protocol; run the existing phase definitions directly.", "",
              "Versions: " + "; ".join(f"{name} {version}" for name, version in sorted(
                  plan.get("package_versions", {}).items())) + ".", "",
              "On this Mac, the external boosting packages use the existing OpenMP library bundled "
              "with scikit-learn. The path below is machine-specific; no system library was replaced.",
              "", "```bash", "cd experiments/exhaustive_search",
              f'export DYLD_LIBRARY_PATH="{LIBOMP_DIRECTORY}${{DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}}"',
              ".venv/bin/python search.py screen --jobs 4",
              ".venv/bin/python search.py nested_0 --jobs 4",
              ".venv/bin/python search.py nested_1 --jobs 4",
              ".venv/bin/python search.py nested_2 --jobs 4",
              ".venv/bin/python assess_search.py --phase screen",
              ".venv/bin/python assess_search.py --phase nested",
              ".venv/bin/python -u -c 'import repeated_validation; repeated_validation.main(2)'",
              ".venv/bin/python report_search.py", "```", "",
              "The compatible procedure uses the existing NumPy/SciPy/scikit-learn stack. CatBoost, "
              "XGBoost, and LightGBM were benchmarked separately because grader support for those "
              "dependencies is unknown. Their local availability does not make a submission using "
              "them grader-compatible.", "", "## Interpretation and artifacts", "",
              "- This is a large **finite** search, not exhaustion of all algorithms, feature "
              "representations, hyperparameters, or continuous ensemble weights.",
              "- All released rows have been reused during historical development. The old 500-row "
              "confirmation partition is now development data. No result here is a fresh hidden-test estimate.",
              "- Large-scale selection makes screen maxima optimistic. Nested validation repeats "
              "selection before accessing outer labels, but historical algorithm development still "
              "used these emails. See scikit-learn's [nested versus non-nested validation explanation]"
              "(https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).",
              "- Repeated folds overlap and are not independent trials. Their variability and win "
              "counts are stability diagnostics, not confidence that a final leaderboard rank is guaranteed.",
              "- At 1% FPR, a few high-scoring ham emails can move TPR materially. The final hidden "
              "test and its distribution remain unknown. The gate concerns both metrics and does "
              "not establish that no untested method could outperform the submission.", "",
              "The complete audit is in [protocol.json](protocol.json), "
              "[screen family summaries](screen_family_summary.json), "
              "[screen selection](selection_screen.json), [nested assessment](nested_assessment.json), "
              "and [repeated results](repeated_results.json). Exact finite blend menus are saved "
              "as `recipes_<phase>_<procedure>.npz`; constituent specifications are in the protocol. "
              "The implementation is in [search.py](search.py), [ensemble_search.py](ensemble_search.py), "
              "[assess_search.py](assess_search.py), and [repeated_validation.py](repeated_validation.py).",
              "", "The final submission decision is recorded in [decision.json](decision.json). "
              "The active `classifySpam.py` remains byte-for-byte identical to the frozen "
              "[baseline](baseline_classifier.py). A self-contained "
              "[experimental candidate](candidate_classifier.py) is retained for inspection, "
              "with [parity and interface checks](candidate_verification.json); it is not "
              "the active submission and did not pass the promotion gate. Raw per-fit caches "
              "remain in the local `results/` directory, which is excluded from version control.",
              ""]
    return "\n".join(lines)


def main(make_chart=True):
    artifacts, phases = load_completed()
    content = render_readme(artifacts, phases, include_plot=make_chart)
    # Nothing is written before all required inputs pass completeness checks.
    if make_chart:
        make_plot(phases["screen"], artifacts["selection_screen"]["procedures"], HERE / "screen_tradeoffs.png")
    (HERE / "README.md").write_text(content)
    print(json.dumps({"report": str(HERE / "README.md"), "gate": promotion_gate(artifacts),
                      "plot": str(HERE / "screen_tradeoffs.png") if make_chart else None}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="Write the completed text audit without the optional plot.")
    arguments = parser.parse_args()
    main(make_chart=not arguments.no_plot)
