#!/usr/bin/env python
"""Figures, table and checks from speed_bench.py's results.   (v1, 2026-10-20)

    python plot_speed.py            # reads $WORK_LUKAS/eval_results/speed/*.json (not *.smoke.json)
    python plot_speed.py --smoke    # the smoke files instead
    python plot_speed.py --dir D    # any results directory

Writes into <dir>/figs/:
  prefill_latency.(pdf|png)   ms per forward pass vs prompt length, log-log
  prefill_speedup.(pdf|png)   teacher time / student time; > 1 = student faster; crossover marked
  prefill_memory.(pdf|png)    peak memory above the weights during the pass
  decode_latency.(pdf|png)    ms per generated token vs context length, batch 1 | batch 8
  decode_cache.(pdf|png)      cache size after the prefill, batch 1
  speed_points.csv            every measured point
  speed_table.md              the thesis table + log-log slopes + crossovers + checks
Colours: teacher in ink, students by model (fixed order); line style = mechanism (KDA solid,
GDN dashed), marker = layout (hybrid o, all-linear s); every line is also labelled at its end.
"""
import argparse
import csv
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W = os.environ.get("WORK_LUKAS", "/leonardo_work/EUHPC_D34_049/WORK_LUKAS")
ORDER = ["teacher", "r1_kda_hybrid", "r2_gdn_hybrid", "r3_kda_only", "r4_gdn_only"]
LABEL = {"teacher": "Teacher (AFM-4.5B)", "r1_kda_hybrid": "KDA hybrid (R1)", "r2_gdn_hybrid": "GDN hybrid (R2)",
         "r3_kda_only": "KDA all-linear (R3)", "r4_gdn_only": "GDN all-linear (R4)"}
SHORT = {"teacher": "Teacher", "r1_kda_hybrid": "R1", "r2_gdn_hybrid": "R2", "r3_kda_only": "R3", "r4_gdn_only": "R4"}
# validated categorical slots 1-4 (dataviz reference palette, light mode; contrast WARN -> direct labels)
COLOR = {"teacher": "#0b0b0b", "r1_kda_hybrid": "#2a78d6", "r2_gdn_hybrid": "#eb6834",
         "r3_kda_only": "#1baf7a", "r4_gdn_only": "#eda100"}
LS = {"teacher": "-", "r1_kda_hybrid": "-", "r2_gdn_hybrid": "--", "r3_kda_only": "-", "r4_gdn_only": "--"}
MK = {"teacher": "D", "r1_kda_hybrid": "o", "r2_gdn_hybrid": "o", "r3_kda_only": "s", "r4_gdn_only": "s"}
INK2, GRID = "#52514e", "#e4e3df"
TEACHER_CTX = 65536
TABLE_T = [2048, 8192, 32768, 65536, 131072]


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": INK2,
        "axes.labelcolor": "#0b0b0b", "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.6, "lines.markersize": 4.5, "savefig.bbox": "tight", "savefig.dpi": 200,
    })


def kfmt(t):
    return f"{t // 1024}k" if t >= 1024 else str(t)


def load(d, smoke):
    res = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        if f.endswith(".smoke.json") != smoke:
            continue
        r = json.load(open(f))
        res[r["name"]] = r
    return {n: res[n] for n in ORDER if n in res}


def series(r, part, b, field):
    out = []
    for k, v in r["points"].items():
        p = k.split("|")
        if len(p) == 3 and p[0] == part and p[1] == f"b{b}" and v.get("status") == "ok":
            out.append((int(p[2][1:]), v[field]))
    return sorted(out)


def first_oom(r, part, b):
    ts = [int(k.split("|")[2][1:]) for k, v in r["points"].items()
          if k.startswith(f"{part}|b{b}|") and v.get("status") == "oom"]
    return min(ts) if ts else None


def xaxis(ax, ts):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ts)
    ax.set_xticklabels([kfmt(t) for t in ts])
    ax.minorticks_off()


def ctx_line(ax):
    ax.axvline(TEACHER_CTX, color=INK2, lw=0.8, ls=":")
    ax.text(TEACHER_CTX, 0.02, "teacher's configured\ncontext (64k) ", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=7, color=INK2)


def end_labels(ax, lines):
    """Direct labels at line ends, nudged apart vertically in display space."""
    pts = []
    for name, (x, y) in lines.items():
        pts.append([ax.transData.transform((x, y))[1], name, x, y])
    pts.sort()
    for i in range(1, len(pts)):
        if pts[i][0] - pts[i - 1][0] < 12:
            pts[i][0] = pts[i - 1][0] + 12
    for yd, name, x, y in pts:
        ytxt = ax.transData.inverted().transform((0, yd))[1]
        ax.annotate(SHORT[name], (x, y), xytext=(x * 1.12, ytxt), fontsize=7.5, color="#0b0b0b",
                    va="center", annotation_clip=False)


def plot_lines(ax, res, part, b, field, scale=1.0, log_y=True, label_ends=True, y0=None):
    allt, ends = set(), {}
    for n, r in res.items():
        s = series(r, part, b, field)
        if not s:
            continue
        ts, ys = zip(*s)
        ys = [y * scale for y in ys]
        allt.update(ts)
        ax.plot(ts, ys, color=COLOR[n], ls=LS[n], marker=MK[n], label=LABEL[n],
                markeredgecolor="white", markeredgewidth=0.6)
        ends[n] = (ts[-1], ys[-1])
        o = first_oom(r, part, b)
        if o:
            ax.annotate(f"OOM at {kfmt(o)}", (ts[-1], ys[-1]), xytext=(-6, 8), textcoords="offset points", ha="right",
                        fontsize=7, color=COLOR[n] if n == "teacher" else "#0b0b0b")
    if log_y:
        ax.set_yscale("log")
    if allt:
        xaxis(ax, sorted(allt))
    if y0 is not None:                  # fix the y range BEFORE placing labels (they use display coordinates)
        ax.set_ylim(bottom=y0)
    if label_ends and ends:
        end_labels(ax, ends)
    return sorted(allt)


def save(fig, d, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"))
    plt.close(fig)


def slope(s, lo=16384, hi=65536):
    pts = [(math.log(t), math.log(y)) for t, y in s if lo <= t <= hi and y > 0]
    if len(pts) < 2:
        return float("nan")
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    return sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)


def crossover(sp):
    """first T where the speed-up crosses 1 from below, log-interpolated; None if never."""
    for (t0, y0), (t1, y1) in zip(sp, sp[1:]):
        if y0 < 1 <= y1:
            f = (0 - math.log(y0)) / (math.log(y1) - math.log(y0))
            return round(math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0))))
    return None if not sp or sp[0][1] < 1 else sp[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{W}/eval_results/speed")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    res = load(a.dir, a.smoke)
    if "teacher" not in res:
        raise SystemExit(f"no teacher results in {a.dir}")
    figs = os.path.join(a.dir, "figs" + ("_smoke" if a.smoke else ""))
    os.makedirs(figs, exist_ok=True)
    style()

    # 1. prefill latency
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    plot_lines(ax, res, "prefill", 1, "ms")
    ctx_line(ax)
    ax.set(xlabel="prompt length T (tokens)", ylabel="forward pass (ms, log scale)",
           title="Prefill: one forward pass, batch 1")
    ax.legend(loc="upper left")
    save(fig, figs, "prefill_latency")

    # 2. speed-up vs the teacher
    tea = dict(series(res["teacher"], "prefill", 1, "ms"))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    allt, ends, cross = set(), {}, {}
    for n, r in res.items():
        if n == "teacher":
            continue
        sp = [(t, tea[t] / y) for t, y in series(r, "prefill", 1, "ms") if t in tea]
        if not sp:
            continue
        ts, ys = zip(*sp)
        allt.update(ts)
        ax.plot(ts, ys, color=COLOR[n], ls=LS[n], marker=MK[n], label=LABEL[n],
                markeredgecolor="white", markeredgewidth=0.6)
        ends[n] = (ts[-1], ys[-1])
        cross[n] = crossover(sp)
    ax.axhline(1.0, color=INK2, lw=0.9)
    ax.text(min(allt), 1.0, "  as fast as the teacher", va="bottom", fontsize=7, color=INK2)
    xaxis(ax, sorted(allt))
    end_labels(ax, ends)
    ctx_line(ax)
    ax.set(xlabel="prompt length T (tokens)", ylabel="speed-up over the teacher (×)",
           title="Prefill speed-up (teacher time / student time)")
    ax.legend(loc="upper left")
    save(fig, figs, "prefill_speedup")

    # 3. prefill memory
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    plot_lines(ax, res, "prefill", 1, "peak_above_weights_mib", scale=1 / 1024)
    ctx_line(ax)
    ax.set(xlabel="prompt length T (tokens)", ylabel="peak memory above the weights (GiB, log)",
           title="Prefill memory, batch 1")
    ax.legend(loc="upper left")
    save(fig, figs, "prefill_memory")

    # 4. decode latency, batch 1 | batch 8 (separate y scales, same x)
    batches = sorted({int(k.split("|")[1][1:]) for r in res.values() for k in r["points"] if k.startswith("decode|")})
    fig, axes = plt.subplots(1, len(batches), figsize=(4.2 * len(batches), 3.6), squeeze=False)
    axes = axes[0]
    for ax, b in zip(axes, batches):
        plot_lines(ax, res, "decode", b, "ms_per_token", log_y=False, y0=0)
        ctx_line(ax)
        ax.set(xlabel="context length T (tokens)", ylabel="ms per generated token",
               title=f"Decode, batch {b}")
    axes[0].legend(loc="lower left", ncol=2)          # the lines sit at 30-55 ms; the bottom is empty
    save(fig, figs, "decode_latency")

    # 5. cache size
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    plot_lines(ax, res, "decode", 1, "cache_mib")
    ctx_line(ax)
    ax.set(xlabel="context length T (tokens)", ylabel="cache after prefill (MiB, log)",
           title="Inference state (KV cache + recurrent state), batch 1")
    ax.legend(loc="upper left")
    save(fig, figs, "decode_cache")

    # CSV of every point
    with open(os.path.join(figs, "speed_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "part", "batch", "T", "status", "ms", "ms_per_token", "wall_ms_per_token",
                    "tok_per_s", "peak_above_weights_mib", "cache_mib", "prefill_with_cache_s"])
        for n, r in res.items():
            for k, v in sorted(r["points"].items()):
                p = k.split("|")
                if len(p) != 3:
                    continue
                w.writerow([n, p[0], p[1][1:], p[2][1:], v.get("status")] +
                           [v.get(c, "") for c in ("ms", "ms_per_token", "wall_ms_per_token", "tok_per_s",
                                                   "peak_above_weights_mib", "cache_mib", "prefill_with_cache_s")])

    # the table
    def cell(n, part, b, field, t, fmt):
        s = dict(series(res[n], part, b, field))
        if t in s:
            return fmt(s[t])
        o = first_oom(res[n], part, b)
        return "OOM" if o and t >= o else "—"

    L = ["# Speed benchmark", "",
         f"GPU: {res['teacher'].get('provenance', {}).get('gpu')}  ·  versions: {res['teacher'].get('provenance', {}).get('versions')}", "",
         "## Prefill, batch 1 — ms per forward pass (speed-up over the teacher)", "",
         "| model | params | " + " | ".join(kfmt(t) for t in TABLE_T) + " | log-log slope 16k–64k | crossover T |",
         "|---|---|" + "---|" * len(TABLE_T) + "---|---|"]
    for n, r in res.items():
        row = [LABEL[n], f"{r.get('params', 0) / 1e9:.2f} B"]
        s = dict(series(r, "prefill", 1, "ms"))
        for t in TABLE_T:
            if t in s:
                row.append(f"{s[t]:.0f}" + (f" ({tea[t] / s[t]:.2f}×)" if n != "teacher" and t in tea else ""))
            else:
                row.append(cell(n, "prefill", 1, "ms", t, str))
        row.append(f"{slope(sorted(s.items())):.2f}")
        row.append("—" if n == "teacher" else (kfmt(cross.get(n)) if cross.get(n) else "never (≤ 128k)"))
        L.append("| " + " | ".join(row) + " |")
    for b in batches:
        L += ["", f"## Decode, batch {b} — ms per generated token (cache MiB)", "",
              "| model | " + " | ".join(kfmt(t) for t in TABLE_T) + " |", "|---|" + "---|" * len(TABLE_T)]
        for n in res:
            row = [LABEL[n]]
            ms = dict(series(res[n], "decode", b, "ms_per_token"))
            cm = dict(series(res[n], "decode", b, "cache_mib"))
            for t in TABLE_T:
                row.append(f"{ms[t]:.1f} ({cm[t]:.0f})" if t in ms else cell(n, "decode", b, "ms_per_token", t, str))
            L.append("| " + " | ".join(row) + " |")
    L += ["", "## Checks", "", "| model | chunked = one-shot prefill: max logit diff (argmax) | next step: max diff (argmax) | logit scale | drift: T=8192 re-measured |",
          "|---|---|---|---|---|"]
    for n, r in res.items():
        c = r["points"].get("chunk_check", {})
        rc = r["points"].get("recheck|prefill|b1|T8192", {})
        p8 = r["points"].get("prefill|b1|T8192", {})
        drift = (f"{100 * (rc['ms'] / p8['ms'] - 1):+.1f} %" if rc.get("status") == "ok" and p8.get("status") == "ok" else "—")
        if c.get("status") == "ok":
            L.append(f"| {LABEL[n]} | {c['prefill_logit_maxdiff']:.3g} ({'same' if c['prefill_argmax_same'] else 'DIFFERS'}) | "
                     f"{c['step_logit_maxdiff']:.3g} ({'same' if c['step_argmax_same'] else 'DIFFERS'}) | {c['logit_scale']:.1f} | {drift} |")
        else:
            L.append(f"| {LABEL[n]} | {c.get('status', '—')} {c.get('error', '')} | | | {drift} |")
    errs = [(n, k, v.get("error")) for n, r in res.items() for k, v in r["points"].items() if v.get("status") == "error"]
    if errs:
        L += ["", "## ⚠️ Errors (not OOM)", ""] + [f"- {n} {k}: {e}" for n, k, e in errs]
    open(os.path.join(figs, "speed_table.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nfigures -> {figs}")


if __name__ == "__main__":
    main()