"""Charts and tables from benchmark results.

Usage: python scripts/make_charts.py

Reads results/summary.csv (written by benchmark.py) and, if it has rows,
results/human_solves.csv (columns: person, task, seed, quarter_turns,
decisions, solved, notes; task is "full" or "stage2".."stage7"). Saves to
results/charts/:

  chart1_tools_vs_raw.png    full tasks: condition A vs B per model -
                             furthest stage (mean), cost per episode, decisions
  chart2_how_far.png         full tasks, condition B: furthest stage completed,
                             stacked per model
  chart3_stage_success.png   stage tasks: success rate per model and stage
  chart4_rewards.png         reward rules vs scripted agents (src/scripted_agents.py)
  chart5_people_vs_models.csv  quarter turns and decisions: people, the
                             scripted method solver and each model, same seeds

Every number used is also printed as a plain-text table. Works on partial
results: a chart with no data yet is drawn as an empty panel saying so.

Models are labelled by their short id; a model run under more than one
reasoning effort gets one entry per effort, e.g. "qwen3.5-9b (none)".
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RESULTS_DIR = ROOT / "results"
STAGE_RANGE = range(2, 8)

# Reference palette (dataviz skill, light mode): categorical slots in fixed
# order, a blue sequential ramp, and text/surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
             "#184f95", "#104281", "#0d366b"]
STAGE_COLORS = ["#d9d8d3", "#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95",
                "#0d366b"]  # stage 0 (none) neutral gray, then stages 1-7 light -> dark
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


# --- reading results ----------------------------------------------------------


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if any((v or "").strip() for v in row.values())]


def entity_labels(rows):
    """{(model, effort): label}. Effort only appears when a model has several."""
    efforts = defaultdict(set)
    for r in rows:
        efforts[r["model"]].add(r.get("reasoning_effort") or "unset")
    labels = {}
    for model, effs in efforts.items():
        short = model.split("/")[-1]
        for e in sorted(effs):
            labels[(model, e)] = f"{short} ({e})" if len(effs) > 1 else short
    return labels


def label_of(row, labels):
    return labels[(row["model"], row.get("reasoning_effort") or "unset")]


def ordered(labels):
    """Entities in a stable order: by model id, then effort."""
    return [labels[k] for k in sorted(labels)]


# --- drawing helpers -------------------------------------------------------------


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(width, height, title):
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=13, x=0.02, ha="left")
    return fig


def _empty(path, title, message):
    fig = _figure(8, 3.5, title)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", color=TEXT_SECONDARY, fontsize=11,
            transform=ax.transAxes)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _save(fig, path):
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def print_table(title, header, rows, out=print):
    out(f"\n{title}")
    if not rows:
        out("  (no data yet)")
        return
    cells = [[str(c) for c in header]] + [[_fmt(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    for i, row in enumerate(cells):
        out("  " + "  ".join(c.ljust(w) if j == 0 else c.rjust(w)
                             for j, (c, w) in enumerate(zip(row, widths))))
        if i == 0:
            out("  " + "  ".join("-" * w for w in widths))


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")  # 0.3, 1.1, 0.0012, 0
    return "" if value is None else str(value)


# --- chart 1: tools vs raw --------------------------------------------------------


def chart1(rows, path, out=print):
    title = "Full tasks: one answer (A) vs method + tools (B)"
    full = [r for r in rows if (r.get("task") or "full") == "full"]
    labels = entity_labels(full)
    groups = defaultdict(list)
    for r in full:
        groups[(label_of(r, labels), r["condition"])].append(r)

    metrics = [("Furthest stage (mean)", lambda rs: mean(_int(r["max_stage"]) for r in rs)),
               ("Cost per episode ($)", lambda rs: mean(_float(r["total_cost_usd"]) for r in rs)),
               ("Decisions (mean)", lambda rs: mean(_int(r["decisions"]) for r in rs))]
    entities = ordered(labels)
    table = []
    for ent in entities:
        for cond in ("A", "B"):
            rs = groups.get((ent, cond))
            if rs:
                table.append([ent, cond, len(rs)] + [round(fn(rs), 4) for _, fn in metrics])
    print_table("Chart 1 - full tasks, condition A vs B",
                ["model", "cond", "episodes"] + [m for m, _ in metrics], table, out)
    if not table:
        _empty(path, title, "No full-task results yet")
        return

    fig = _figure(4.2 * len(metrics), 4.2, title)
    width = 0.38
    for m, (name, fn) in enumerate(metrics):
        ax = fig.add_subplot(1, len(metrics), m + 1)
        _style(ax)
        for c, cond in enumerate(("A", "B")):
            xs, ys = [], []
            for i, ent in enumerate(entities):
                rs = groups.get((ent, cond))
                if rs:
                    xs.append(i + (c - 0.5) * width)
                    ys.append(fn(rs))
            bars = ax.bar(xs, ys, width=width - 0.04, color=SERIES[c],
                          label=f"{cond}: {'one answer' if cond == 'A' else 'method + tools'}")
            fmt = "{:.4f}" if name.startswith("Cost") else "{:.1f}"
            ax.bar_label(bars, labels=[fmt.format(y) for y in ys], fontsize=7,
                         color=TEXT_SECONDARY, padding=2)
        ax.set_title(name, color=TEXT_PRIMARY, fontsize=10, loc="left")
        ax.set_xticks(range(len(entities)))
        ax.set_xticklabels(entities, rotation=30, ha="right")
        if name.startswith("Furthest"):
            ax.set_ylim(0, 7.5)
        if m == 0:
            ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper left")
    _save(fig, path)


# --- chart 2: how far condition B gets ----------------------------------------------


def chart2(rows, path, out=print):
    title = "Full tasks, condition B: furthest stage completed"
    full_b = [r for r in rows if (r.get("task") or "full") == "full" and r["condition"] == "B"]
    labels = entity_labels(full_b)
    counts = defaultdict(lambda: [0] * 8)
    for r in full_b:
        counts[label_of(r, labels)][min(max(_int(r["max_stage"]), 0), 7)] += 1
    entities = ordered(labels)
    table = [[ent] + counts[ent] + [sum(counts[ent])] for ent in entities]
    print_table("Chart 2 - full tasks, condition B: episodes by furthest stage completed",
                ["model"] + [f"stage {s}" for s in range(8)] + ["episodes"], table, out)
    if not table:
        _empty(path, title, "No condition B full-task results yet")
        return

    fig = _figure(max(6, 1.3 * len(entities) + 3), 4.6, title)
    ax = fig.add_subplot(111)
    _style(ax)
    bottoms = [0] * len(entities)
    for stage in range(8):
        heights = [counts[ent][stage] for ent in entities]
        if not any(heights):
            continue
        name = "0 (none)" if stage == 0 else f"{stage}" + (" (solved)" if stage == 7 else "")
        ax.bar(range(len(entities)), heights, bottom=bottoms, width=0.55,
               color=STAGE_COLORS[stage], edgecolor=SURFACE, linewidth=2, label=name)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_xticks(range(len(entities)))
    ax.set_xticklabels(entities, rotation=20, ha="right")
    ax.set_ylabel("episodes", color=TEXT_SECONDARY, fontsize=9)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(title="stage", frameon=False, fontsize=8, title_fontsize=8, labelcolor=TEXT_SECONDARY,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, path)


# --- chart 3: stage-task success ---------------------------------------------------


def chart3(rows, path, out=print):
    title = "Stage tasks: success rate (tasks reaching their target)"
    stage_rows = [r for r in rows if (r.get("task") or "full").startswith("stage")]
    labels = entity_labels(stage_rows)
    cells = defaultdict(lambda: [0, 0])  # (entity, stage) -> [successes, tasks]
    for r in stage_rows:
        key = (label_of(r, labels), _int(r["task"][len("stage"):]))
        cells[key][1] += 1
        cells[key][0] += _bool(r["target_reached"])
    entities = ordered(labels)
    table = []
    for ent in entities:
        row = [ent]
        for s in STAGE_RANGE:
            k, n = cells.get((ent, s), (0, 0))
            row.append(f"{k}/{n}" if n else "-")
        table.append(row)
    print_table("Chart 3 - stage tasks: successes / tasks",
                ["model"] + [f"stage {s}" for s in STAGE_RANGE], table, out)
    if not table:
        _empty(path, title, "No stage-task results yet")
        return

    fig = _figure(8.5, 1.0 + 0.55 * len(entities) + 1.2, title)
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    for y, ent in enumerate(entities):
        for x, s in enumerate(STAGE_RANGE):
            k, n = cells.get((ent, s), (0, 0))
            if not n:
                ax.add_patch(plt.Rectangle((x + 0.03, y + 0.03), 0.94, 0.94, color=SURFACE,
                                           ec=GRID, lw=0.8))
                ax.text(x + 0.5, y + 0.5, "-", ha="center", va="center", color=TEXT_SECONDARY)
                continue
            rate = k / n
            ax.add_patch(plt.Rectangle((x + 0.03, y + 0.03), 0.94, 0.94, color=cmap(rate)))
            ink = "#ffffff" if rate >= 0.5 else TEXT_PRIMARY
            ax.text(x + 0.5, y + 0.5, f"{rate:.0%}\n{k}/{n}", ha="center", va="center",
                    color=ink, fontsize=8)
    ax.set_xlim(0, len(STAGE_RANGE))
    ax.set_ylim(len(entities), 0)
    ax.set_xticks([x + 0.5 for x in range(len(STAGE_RANGE))])
    ax.set_xticklabels([f"stage {s}" for s in STAGE_RANGE])
    ax.set_yticks([y + 0.5 for y in range(len(entities))])
    ax.set_yticklabels(entities)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.text(0, -0.25, "cell: success rate, successes/tasks; colour darker = higher rate",
            color=TEXT_SECONDARY, fontsize=8, transform=ax.transData)
    _save(fig, path)


# --- chart 4: reward rules -----------------------------------------------------------


def reward_table():
    from scripted_agents import AGENTS, RULES, run_episode

    names = {"farming_agent": "farming (D, D' x10)",
             "method_sequence_agent": "correct stage-7 solve"}
    return [[names.get(a, a)] + [round(sum(run_episode(agent, rule())), 4) for rule in RULES.values()]
            for a, agent in AGENTS.items()], list(RULES)


def chart4(path, out=print):
    title = "Reward rules: total reward per scripted episode"
    table, rules = reward_table()
    print_table("Chart 4 - reward rules (from src/scripted_agents.py)", ["agent"] + rules, table, out)

    fig = _figure(7.5, 4.2, title)
    ax = fig.add_subplot(111)
    _style(ax)
    width = 0.8 / len(rules)
    for j, rule in enumerate(rules):
        xs = [i + (j - (len(rules) - 1) / 2) * width for i in range(len(table))]
        ys = [row[1 + j] for row in table]
        bars = ax.bar(xs, ys, width=width - 0.03, color=SERIES[j], label=rule)
        ax.bar_label(bars, labels=[f"{y:.1f}" for y in ys], fontsize=8, color=TEXT_SECONDARY,
                     padding=2)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels([row[0] for row in table])
    ax.set_ylabel("total reward", color=TEXT_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    _save(fig, path)


# --- chart 5: people vs models (CSV) --------------------------------------------------


CHART5_FIELDS = ["who", "kind", "task", "seed", "condition", "quarter_turns", "decisions",
                 "target_reached"]


def people_vs_models(rows, human_rows):
    """Rows for chart5: people, the method solver and each model on shared seeds."""
    from method_solver import solve
    from stages import stages_complete
    from tasks import make_stage_task

    out = []
    for h in human_rows:
        out.append({"who": h["person"], "kind": "person", "task": h["task"], "seed": _int(h["seed"]),
                    "condition": "-", "quarter_turns": _int(h["quarter_turns"], None),
                    "decisions": _int(h["decisions"], None), "target_reached": _bool(h["solved"])})
    labels = entity_labels(rows)
    for r in rows:
        out.append({"who": label_of(r, labels), "kind": "model", "task": r.get("task") or "full",
                    "seed": _int(r["seed"]), "condition": r["condition"],
                    "quarter_turns": _int(r["quarter_turns"]), "decisions": _int(r["decisions"]),
                    "target_reached": _bool(r["target_reached"])})

    # The scripted solver on every stage task anyone attempted (seeds 1-5 if none
    # yet). It doesn't do stage 1, so it has no full-task rows.
    seeds = {(t["task"], t["seed"]) for t in out if str(t["task"]).startswith("stage")}
    if not seeds:
        seeds = {(f"stage{s}", seed) for s in STAGE_RANGE for seed in range(1, 6)}
    for task, seed in sorted(seeds, key=lambda ts: (_int(ts[0][5:]), ts[1])):
        stage = _int(task[len("stage"):])
        start = make_stage_task(stage, seed)
        actions, final = solve(start, stage)
        turns = sum(1 if name == "turn" else _alg_turns(args) for name, args in actions)
        out.append({"who": "method solver", "kind": "scripted", "task": task, "seed": seed,
                    "condition": "B", "quarter_turns": turns, "decisions": len(actions),
                    "target_reached": stages_complete(final) >= stage})

    kind_order = {"person": 0, "scripted": 1, "model": 2}
    out.sort(key=lambda t: (_task_order(t["task"]), t["seed"], kind_order[t["kind"]], t["who"],
                            t["condition"]))
    return out


def _alg_turns(args):
    from algorithms import ALGORITHMS, parse

    return len(parse(ALGORITHMS[args["name"]])) * args.get("times", 1)


def _task_order(task):
    return 0 if task == "full" else _int(str(task)[len("stage"):])


def chart5(rows, human_rows, path, out=print):
    table = people_vs_models(rows, human_rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHART5_FIELDS)
        writer.writeheader()
        writer.writerows(table)
    print_table("Chart 5 - people vs models (quarter turns, decisions)", CHART5_FIELDS,
                [[t[k] for k in CHART5_FIELDS] for t in table], out)


# --- main --------------------------------------------------------------------------------


def make_charts(results_dir=RESULTS_DIR, out=print):
    results_dir = Path(results_dir)
    charts = results_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    rows = read_rows(results_dir / "summary.csv")
    human = read_rows(results_dir / "human_solves.csv")
    out(f"{len(rows)} benchmark episodes in summary.csv; {len(human)} human solves")

    chart1(rows, charts / "chart1_tools_vs_raw.png", out)
    chart2(rows, charts / "chart2_how_far.png", out)
    chart3(rows, charts / "chart3_stage_success.png", out)
    chart4(charts / "chart4_rewards.png", out)
    chart5(rows, human, charts / "chart5_people_vs_models.csv", out)
    out(f"\nsaved to {charts}")
    return charts


if __name__ == "__main__":
    make_charts()
