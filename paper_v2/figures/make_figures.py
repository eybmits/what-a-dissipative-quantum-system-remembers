#!/usr/bin/env python3
"""Figures 2-4, the dark-state figure and the appendix figures (mixing law, other sizes) from
paper_v2/data/snapshot.json (written by export_snapshot.py). Recall curves are drawn smoothed over neighbouring delays
(smooth, curve); every number in the text uses the unsmoothed recall.

Figures across the page hold one row of panels, as large as the page allows, with their keys and colour scale inside
the panels, in space the data leave free; a key that several panels share sits in a row under them. The one-column
figures carry their keys inside the panels too. Every colour, marker and line style is named in its own figure.
Aligned spines, letters and axis labels, no titles, one set of sizes for lines and markers, all text at the 10 pt of
the captions, and no text on the data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import ticker
from matplotlib.collections import LineCollection
from scipy.interpolate import Akima1DInterpolator

import style as st

HERE = Path(__file__).resolve().parent
SNAP = json.loads((HERE.parent / "data" / "snapshot.json").read_text())
ORDER = ("separate", "exchange", "gainloss", "dephasing", "pair", "signed", "groups", "shared")
FAR_MAX = 4096.0                      # largest loss rate shown for the slow modes
SCALED = (0, (3.2, 1.1, 0.9, 1.1))    # gentle writing: the input amplitude follows the loss
FEATURES = (0, (4, 2))                # the limit set by the number of features


def families():
    return [f for f in ORDER if f in SNAP["memory"]]


def stack(rows):
    return np.array([[np.nan if v is None else v for v in row] for row in rows], float)


def smooth(C, w=0.2):
    """Recall at each delay tau averaged over the neighbouring delays with triangular weights that reach zero at
    w tau. The window is symmetric and is cut short at the last delay, so the average removes the noise of the finite
    test set without shifting the curve; delays up to 1/w are left as they are, which keeps sharp drops sharp."""
    C = np.asarray(C, float)
    n = len(C)
    out = np.empty(n)
    for i in range(n):
        h = min(w * (i + 1), n - 1 - i)
        if h <= 1:
            out[i] = C[i]
            continue
        j = np.arange(-int(np.ceil(h)) + 1, int(np.ceil(h)))
        weight = 1 - np.abs(j) / h
        out[i] = weight @ C[i + j] / weight.sum()
    return out


def curve(C, w=0.2):
    """The smoothed recall as a smooth line through the delays 1, 2, ... on a logarithmic delay axis."""
    y = smooth(C, w)
    x = np.log(np.arange(1, len(y) + 1))
    xf = np.linspace(0.0, x[-1], 600)
    return np.exp(xf), Akima1DInterpolator(x, y, method="makima")(xf)


def curve_band(rows, w=0.2):
    """Mean and 16th-84th percentiles over reservoirs of the smoothed recall, as smooth lines on a log delay axis."""
    S = np.array([smooth(r, w) for r in rows])
    x = np.log(np.arange(1, S.shape[1] + 1))
    xf = np.linspace(0.0, x[-1], 600)
    f = lambda y: Akima1DInterpolator(x, y, method="makima")(xf)
    return np.exp(xf), f(S.mean(axis=0)), f(np.percentile(S, 16, axis=0)), f(np.percentile(S, 84, axis=0))


def recall_band(ax, rows, color, lw=None, alpha=0.18):
    """Recall of several reservoirs: mean line and 16th-84th percentile band in the colour of the line."""
    x, m, lo, hi = curve_band(rows)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)
    ax.plot(x, m, color=color, lw=st.LW if lw is None else lw)


def recall_map(ax, fam):
    """Recall (colour) against loss rate and delay, mean over the reservoirs, smoothed over neighbouring delays;
    dashed: the delay below which half of the memory lies. Between two computed loss rates the recall profile is
    shifted along the delay by the change of that half-memory delay and interpolated, so that sharp edges move smoothly."""
    rec = SNAP["memory"][fam]
    g = np.log10(rec["curve_gamma"])
    curves = np.array([smooth(c) for c in rec["curve_mean"]])            # (loss rates, delays)
    top = np.log10(300)
    tau = np.log10(np.arange(1, curves.shape[1] + 1))
    profiles = [Akima1DInterpolator(tau, c, method="makima") for c in curves]
    half = []
    for c in curves:
        cum = np.cumsum(np.clip(c, 0, None))
        half.append(np.log10(np.interp(0.5 * cum[-1], cum, np.arange(1, len(c) + 1))))
    half = np.array(half)
    gx = np.linspace(g[0], g[-1], 240)
    ty = np.linspace(0, top, 240)
    at = lambda k, y: profiles[k](np.clip(y, 0, top))
    image = np.empty((len(ty), len(gx)))
    for j, x in enumerate(gx):
        k = min(int(np.searchsorted(g, x, side="right")) - 1, len(g) - 2)
        f = (x - g[k]) / (g[k + 1] - g[k])
        shift = half[k + 1] - half[k]
        image[:, j] = (1 - f) * at(k, ty - f * shift) + f * at(k + 1, ty + (1 - f) * shift)
    im = ax.imshow(np.clip(image, 0, 1), origin="lower", aspect="auto", cmap=st.RECALL, vmin=0, vmax=1,
                   extent=(g[0], g[-1], 0, top), interpolation="bilinear", rasterized=True)
    ax.plot(gx, np.clip(np.interp(gx, g, half), 0, top), color="white", lw=st.LW, ls=(0, (3, 1.5)))
    st.log_on_linear(ax, "x", g[0], g[-1], "plain")
    st.log_on_linear(ax, "y", 0, top, "plain")
    ax.tick_params(which="both", direction="out", top=False, right=False)
    ax.set_xlim(g[0], g[-1])
    ax.set_ylim(0, top)
    # ticks on the top edge: the computed loss rates, between which the map interpolates
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(g)
    sec.xaxis.set_minor_locator(ticker.NullLocator())
    sec.tick_params(direction="out", length=2.4, width=0.6, labeltop=False, color=st.INK)
    sec.spines["top"].set_visible(False)
    ax.set_xlabel(r"Loss rate $\gamma$")
    ax.set_ylabel(r"Delay $\tau$")
    return im


def recall_axes(ax, top):
    """Delay axis of a recall panel."""
    ax.set_xscale("log")
    st.decades(ax, "x")
    ax.set_xlim(1, top)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"Delay $\tau$")
    ax.set_ylabel(r"Recall $C_\tau$")


# ----------------------------------------------------------------------------------------------- Figure 2
def gentle_line(design, rule):
    """Memory along a gentle-writing line from the sampling runs: loss rates, rows over reservoirs."""
    recs = sorted((r for r in SNAP.get("sampling", []) if r["design"] == design and r["rule"] == rule),
                  key=lambda r: r["gamma"])
    n = min((len(r["exact"]) for r in recs), default=0)
    return np.array([r["gamma"] for r in recs]), np.array([r["exact"][:n] for r in recs]).T


def decay_rates(ax):
    """Every decay rate of one reservoir at the mean input with one shared channel, each sorted rate drawn as a thin
    line along the loss: the dark modes, one for each protected operator besides the stationary state, from dark to
    lighter blue, the bright modes above them in pale blue; dashed: the slowest mode of local loss."""
    rates = SNAP["rates"]
    fl, loc = rates["flow"], rates["separate"]
    g, R = np.array(fl["gamma"]), np.array(fl["rates"])           # R: (loss rates, sorted modes)
    n = rates["shared"]["protected"] - 1
    base = st.FAMILY["shared"]
    ax.add_collection(LineCollection([np.c_[g, R[:, k]] for k in range(n, R.shape[1])], colors=st.tone(base, 0.72),
                                     lw=0.3, alpha=0.35, rasterized=True, zorder=1))
    ax.add_collection(LineCollection([np.c_[g, R[:, k]] for k in range(n)], lw=0.45, rasterized=True, zorder=2,
                                     colors=[st.tone(base, 0.35 * k / (n - 1)) for k in range(n)]))
    ax.plot(loc["gamma"], loc["slowest"], color=st.FAMILY["separate"], ls=st.DASH["separate"], zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    st.decades(ax, "x", every=2)
    st.decades(ax, "y", labels="power", every=2)
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(3e-4, 3e3)
    ax.set_xlabel(r"Loss rate $\gamma$")
    ax.set_ylabel(r"Decay rate")
    # the two bands named in the empty space above and below them
    ax.text(0.14, 150, "bright", color=st.tone(st.FAMILY["shared"], 0.35), fontsize=st.TXT, va="baseline")
    ax.text(0.14, 8e-4, "dark", color=st.FAMILY["shared"], fontsize=st.TXT, va="baseline")


def fig2():
    """The shape sets the direction of the clock: recall at weak and at strong loss, memory against loss and the decay
    rates in one row. The two recall maps share their delay axis and carry the colour scale in an empty corner; the
    memory panel is the widest and carries the key of the shapes above the limit set by the features."""
    fig = st.wide_figure(1, hp=st.HP_ROW)
    a, b, c, d = (st.wide_place(fig, k) for k in range(4))

    # a, b: recall against loss rate and delay, local loss and one shared channel; dashed: delay below which half of
    # the memory lies
    im = recall_map(a, "separate")
    recall_map(b, "shared")
    b.set_ylabel("")
    b.tick_params(axis="y", which="both", labelleft=False)

    # c: memory against loss for every shape, the four that carry the story in colour and the others in grey;
    # circles: gentle writing
    for fam in families():
        rec = SNAP["memory"][fam]
        x = np.array(rec["gamma"])
        # beyond gamma = 64 the recall of one and two shared channels reaches past the 300 delays summed
        inside = x <= (256 if fam in ("signed", "pair") else 64)
        if fam in st.MAIN:
            st.band(c, x[inside], stack(rec["recall"])[:, inside], st.FAMILY[fam], ls=st.DASH[fam])
        else:
            c.plot(x[inside], np.nanmean(stack(rec["recall"])[:, inside], axis=0), color=st.OTHER, lw=st.LW_THIN,
                   ls=st.DASH[fam], zorder=1.5)
    sc = SNAP["review"]["scaled"]["1"]
    st.band(c, np.array(sc["gamma"]), stack(sc["recall"]), st.FAMILY["signed"], ls=SCALED, marker="o", ms=st.MS,
            alpha=0.12)
    gx, rows = gentle_line("separate", "gentle_weak")
    if len(gx):
        gx = np.r_[gx, 1.0]
        one = stack(SNAP["memory"]["separate"]["recall"])[:rows.shape[0], SNAP["memory"]["separate"]["gamma"].index(1.0)]
        st.band(c, gx, np.c_[rows, one], st.FAMILY["separate"], ls=SCALED, marker="o", ms=st.MS, alpha=0.12)
    c.axhline(45, color=st.INK3, lw=st.LW_GUIDE, ls=FEATURES)
    # the limit named under its line, where the data leave room; the key of the shapes sits above it
    c.annotate("45 features", xy=(0.25, 45), xycoords="data", xytext=(0, -10.5),
               textcoords="offset points", ha="left", va="baseline", color=st.INK3, fontsize=st.TXT)
    c.set_xscale("log")
    st.decades(c, "x", every=2)
    c.set_xlim(1 / 40, 400)
    c.set_xlabel(r"Loss rate $\gamma$")
    c.set_ylim(0, 78)                       # room above the limit for the three rows of the key
    c.set_yticks([0, 15, 30, 45])
    c.set_ylabel(r"Capacity $\sum_\tau C_\tau$")

    # d: decay rates, bright and dark modes of the equal-phase channel and the slowest mode of independent decay
    decay_rates(d)

    st.fit_row(fig, (a, b, c, d), (70, 70, 152, 80), joined=(b,))
    for ax, name in zip((a, b, c, d), "abcd"):
        st.letter(ax, name)
    st.align(fig, rows=[(a, b, c, d)])
    st.inset_colour_key(a, im, r"$C_\tau$", [0, 0.5, 1])
    # under the colour scale, in the empty top of a: the white dashed line of both maps
    st.inkey(a, [(st.line_handle("white", ls=(0, (3, 1.5)), under=st.RECALL(0.6)), "median")], "upper right",
             pad=(4.5, 26.0))
    shape = lambda f: (st.line_handle(st.FAMILY[f], ls=st.DASH[f]), st.NAME[f])
    st.inkey(c, [shape("separate"), shape("signed"),
                 shape("shared"), (st.line_handle(st.OTHER, lw=st.LW_THIN), "other channels"),
                 shape("dephasing"), (st.line_handle(st.INK3, ls=SCALED, marker="o"), "gentle writing")],
             "upper left", ncol=2, pad=(3.5, 5.0), handle=8.0, gap=2.5, sep=4.0)
    st.save(fig, HERE / "fig2_clock.pdf")


# ----------------------------------------------------------------------------------------------- Figure 3
def qubit(k):
    chi = k ** 2 / ((k + 2) * (k + 1))
    size = 4 * k / ((k + 2) * (3 * k + 2))
    single = 1 - chi
    order = (1 + chi) / (1 - chi)
    return size, single, order


def fig3():
    """The price of a long memory and how the input pays it: the collapse of the time to forget, the write along and
    across the phases, the five-spin order against the mixing law and the qubit, in one row; the keys sit inside the
    panels (a names its markers, its colours are those of c)."""
    fig = st.wide_figure(1, hp=st.HP_ROW)
    a, b, c, d = (st.wide_place(fig, k) for k in range(4))
    dashed = (0, (3, 1.5))

    # a: the collapse. eta^2 t_f against gamma/eta^2 (local loss) and gamma eta^2 (opposite phases), one marker per
    # amplitude
    ck = SNAP["clock"]
    marks = ("o", "s", "^", "D")
    for fam, sign in (("separate", -1), ("signed", 1)):
        amps = sorted(ck[fam], key=float, reverse=True)
        for i, amp in enumerate(amps):
            a_ = float(amp)
            rec = ck[fam][amp]
            g = np.array(rec["gamma"])
            m = np.nanmean(stack(rec["forget"]), axis=0)
            col = st.tone(st.FAMILY[fam], 0.15 * i)
            x, y = g * a_ ** (2 * sign), a_ * a_ * m
            inside = (x >= 2e-3) & (x <= 5e3) & (y >= 1e-3) & (y <= 1e3)
            a.plot(x[inside], y[inside], ls="none", marker=marks[i], ms=2.9, mfc="white", mec=col, mew=0.8,
                   clip_on=False)
    a.set_xscale("log")
    a.set_yscale("log")
    st.decades(a, "x", labels="power", every=2)
    st.decades(a, "y", labels="power", every=2)
    a.set_xlim(2e-3, 5e3)
    a.set_ylim(1e-3, 1e4)                   # room above the points for the key of the amplitudes
    a.set_xlabel(r"$\gamma/\eta^2$ or $\gamma\eta^2$")
    a.set_ylabel(r"$\eta^2\,t_{\mathrm f}$")

    # b: write per input into the dark states of the equal-phase channel, for the input along the phases (uniform)
    # and across them (cosine pattern); dotted: 1/gamma
    W = SNAP["write"]
    g = np.array(W["gamma"])
    for key, col in (("along", st.FAMILY["shared"]), ("across", st.FAMILY["signed"])):
        st.band(b, g, np.array(W[key]), col, marker="o", ms=st.MS, alpha=0.15)
    along = np.nanmean(np.array(W["along"]), axis=0)
    gg = np.geomspace(16, 160, 10)
    b.plot(gg, 0.5 * along[-1] * g[-1] / gg, color=st.INK3, lw=st.LW_GUIDE, ls=(0, (1, 1.6)), zorder=0)
    b.set_xscale("log")
    b.set_yscale("log")
    st.decades(b, "x", every=2)
    st.decades(b, "y", labels="power")
    b.set_xlim(0.15, 250)
    b.set_ylim(2e-4, 1)                     # room under the curves for the key
    b.set_xlabel(r"Loss rate $\gamma$")
    b.set_ylabel(r"Write per input")

    # c: five spins, mean order against the mixing law; room above the points for the key
    five_law(c)
    c.set_ylim(0.8, 60)

    # d: one qubit in the limit of slow relaxation
    k = np.geomspace(0.01, 100, 400)
    size, single, order = qubit(k)
    d.plot(k, size / size.max(), color=st.GENTLE)
    d.plot(k, single, color=st.INK2, ls=dashed)
    # the largest signal, up to the curve; the key sits above the curves
    d.plot([2 / np.sqrt(3)] * 2, [0, 1], color=st.GENTLE, lw=st.LW_GUIDE, ls=(0, (1, 1.6)))
    d.set_xscale("log")
    st.decades(d, "x")
    d.set_xlim(0.02, 50)
    d.set_ylim(0, 1.4)
    d.set_yticks([0, 0.5, 1])
    d.set_xlabel(r"$\theta^2/\epsilon$")
    d.set_ylabel(r"Signal, share")

    st.fit_row(fig, (a, b, c, d), (80, 76, 118, 78))
    for ax, name in zip((a, b, c, d), "abcd"):
        st.letter(ax, name)
    st.align(fig, rows=[(a, b, c, d)])
    st.inkey(a, [(st.marker_handle(st.INK2, marker=m, ms=2.9, mfc="white", mec=st.INK2, mew=0.8), label)
                 for m, label in zip(marks, (r"$\eta=1$", r"$1/2$", r"$1/4$", r"$1/8$"))],
             "upper right", ncol=2, handle=6.0, gap=2.5)
    st.inkey(b, [(st.line_handle(st.FAMILY["shared"], marker="o"), "aligned"),
                 (st.line_handle(st.FAMILY["signed"], marker="o"), "not aligned"),
                 (st.line_handle(st.INK3, ls=(0, (1, 1.6)), lw=st.LW_GUIDE), r"$1/\gamma$")], "lower left",
             pad=(3.0, 5.0))
    st.inkey(c, [(st.marker_handle(st.FAMILY["separate"], mfc=st.FAMILY["separate"], mec=st.FAMILY["separate"],
                                   mew=0.9), "independent"),
                 (st.marker_handle(st.FAMILY["signed"], mfc=st.FAMILY["signed"], mec=st.FAMILY["signed"], mew=0.9),
                  "opposite phases"),
                 (st.marker_handle(st.INK3, mfc="white", mec=st.INK3, mew=0.9), "gentle writing"),
                 (st.line_handle(st.INK3, ls=(0, (4, 2)), lw=st.LW_GUIDE), "mixing law")], "upper left")
    st.inkey(d, [(st.line_handle(st.GENTLE), "signal"),
                 (st.line_handle(st.INK2, ls=dashed), "single inputs")], "upper right")
    st.save(fig, HERE / "fig3_price.pdf")


# ------------------------------------------------------------------------------- the dark-state figure
def fig_darkmodel():
    """The dark-state model and the erasure test, two panels across one column that share the recall
    axis, as tall as the panels of Figs. 2 and 3, each with its key in the room above the curves."""
    fig = st.wide_figure(1, width=st.COLUMN, hp=86.0)
    a, b = st.wide_place(fig, 0), st.wide_place(fig, 1)

    # a: lines are the full reservoir, circles the dark-state model
    red = SNAP["memory"]["shared"]["reduced"]
    pick = np.unique(np.round(np.geomspace(1, 200, 9)).astype(int)) - 1
    light = st.tone(st.FAMILY["shared"], 0.5)
    for gkey, col in (("4", light), ("64", st.FAMILY["shared"])):
        recall_band(a, red[gkey]["full_rows"], col)
        a.plot(pick + 1, smooth(red[gkey]["reduced"])[pick], "o", ms=st.MS, mfc="white", mec=col, mew=0.9,
               clip_on=False)
    recall_axes(a, 300)

    # b: mean of the reservoirs
    curves = SNAP["removal"]["16"]["curves"]
    rot = st.tone(st.FAMILY["shared"], 0.5)
    for field, col in (("removed", st.ERASED), ("rotated", rot), ("intact", st.FAMILY["shared"])):
        recall_band(b, curves[field], col, alpha=0.14)
    recall_axes(b, 100)
    for ax in (a, b):
        ax.set_ylim(0, 1.7)                 # room above the curves for the keys, three rows in b
    b.set_ylabel("")
    b.tick_params(axis="y", which="both", labelleft=False)

    # a is the wider panel: its key is the wider one
    st.fit_row(fig, (a, b), (11, 9), joined=(b,))
    for ax, name in zip((a, b), "ab"):
        st.letter(ax, name)
    st.align(fig, rows=[(a, b)])
    st.inkey(a, [(st.line_handle(light), r"$\gamma=4$"), (st.line_handle(st.FAMILY["shared"]), r"$\gamma=64$")],
             "upper right", ncol=2)
    st.inkey(a, [(st.marker_handle(st.INK2, mfc="white", mec=st.INK2, mew=0.9), "dark-state model")],
             "upper right", pad=(4.5, 5.0 + st.KEY_ROW))
    st.inkey(b, [(st.line_handle(st.FAMILY["shared"]), "intact"), (st.line_handle(rot), "copy erased"),
                 (st.line_handle(st.ERASED), "dark erased")], "upper right")
    st.save(fig, HERE / "figD_dark.pdf")


# ------------------------------------------------------------------------------------- five-spin order and law
def five_rows(fam, full):
    """Entries of the five-spin order runs along the full amplitude (full=True) or the gentle line, by loss rate."""
    rows = [r for r in SNAP.get("five_orders", []) if r["design"] == fam and "law_mean" in r]
    gentle = lambda g: min(1.0, g ** (-0.5 if fam == "signed" else 0.5))
    sel = [r for r in rows if (r["eta"] == 1.0) == full and (full or abs(r["eta"] - gentle(r["gamma"])) < 1e-12)]
    return sorted(sel, key=lambda r: r["gamma"])


def five_law(ax):
    """Mean order of the five-spin memory against the mixing law's s t_mem: full amplitude (filled, along the loss
    rate) and gentle writing (open); dashed: the law."""
    for fam in ("separate", "signed"):
        col = st.FAMILY[fam]
        for full in (True, False):
            sel = five_rows(fam, full)
            if not sel:
                continue
            x = np.array([np.mean(r["law_mean"]) for r in sel])
            y = np.array([np.mean(r["mean_order"]) for r in sel])
            assert np.all((x > 4e-4) & (x < 80) & (y > 0.8) & (y < 45)), "a point lies outside panel b"
            if full:
                ax.plot(x, y, color=col, lw=st.LW_THIN, zorder=1)
            ax.plot(x, y, "o", ms=st.MS, mfc=col if full else "white", mec=col, mew=0.9, ls="none", zorder=2,
                    clip_on=False)
    t = np.geomspace(0.5, 100, 50)
    ax.plot(t, t, color=st.INK3, lw=st.LW_GUIDE, ls=(0, (4, 2)), zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    st.decades(ax, "x", labels="plain", every=2)
    st.decades(ax, "y", labels="plain")
    ax.set_xlim(4e-4, 80)
    ax.set_ylim(0.8, 45)
    ax.set_xlabel(r"$s\,t_{\mathrm{mem}}$")
    ax.set_ylabel(r"Mean $D$")


def five_shape(ax):
    """Share of the five-spin capacity carried by products of more than K inputs against K/(s t_mem), at the
    longest memory of each shape at the full amplitude, and the exponential law."""
    for fam in ("separate", "signed"):
        sel = five_rows(fam, True)
        if not sel:
            continue
        r = sel[0] if fam == "separate" else sel[-1]
        scale = float(np.mean(r["law_mean"]))
        for edges, binned in zip(r["edges"], r["binned"]):
            e = np.array(edges[:-1], float)
            surv = 1.0 - np.cumsum(binned)                 # share beyond the bin that starts at e
            ax.plot((e + np.diff(edges) - 1) / scale, np.clip(surv, 1e-6, 1), color=st.FAMILY[fam], lw=st.LW_THIN,
                    alpha=0.8)
    x = np.linspace(0, 4, 100)
    ax.plot(x, np.exp(-x), color=st.INK3, lw=st.LW_GUIDE, ls=(0, (4, 2)))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$K/(s\,t_{\mathrm{mem}})$")
    ax.set_ylabel(r"Capacity beyond $K$")


# ----------------------------------------------------------------------------------------------- Figure A1
def fig_law():
    """One reservoir at three reset probabilities against the mixing law exp(-K/(s t_mem)) = exp(-beta p K): lines
    the exact moments, circles the law; one key names the reset probabilities, the key left of it lines and circles."""
    fig, a = st.column_figure(height=56.0)
    law = SNAP["law"]
    entries = []
    for p, f in (("0.003", 0.0), ("0.01", 0.28), ("0.03", 0.55)):
        y = np.array(law["one"][p])
        K = np.arange(len(y))
        col = st.tone(st.INK, f)
        a.plot(K, y, color=col)
        step = max(1, int(round(0.3 / (law["beta"][p] * float(p)))))
        Ks = K[::step]
        Ks = Ks[Ks <= 300]
        a.plot(Ks, np.exp(-law["beta"][p] * float(p) * Ks), "o", ms=st.MS, mfc="white", mec=col, clip_on=False)
        entries.append((st.line_handle(col, marker="o"), rf"$p={float(p) * 100:g}\%$"))
    a.set_xlim(0, 300)
    a.set_ylim(0, 1.2)                      # room above the curves for the keys
    a.set_yticks([0, 0.5, 1])
    a.set_xlabel(r"Inputs mixed, $K$")
    a.set_ylabel(r"Share beyond $K$")
    st.align(fig, rows=[(a,)])
    st.inkey(a, entries, "upper right")
    left = min(box[0] for _, box in fig.key_boxes[-len(entries):])
    W, Hf = st._pts(fig)
    st.key(fig, left - st.KEY_SEP, [(st.line_handle(st.INK2), "exact"),
                                    (st.marker_handle(st.INK2, mfc="white", mec=st.INK2, mew=0.8), "mixing law")],
           y=a.get_position().y1 * Hf - 5.0 - st.ASC, right=True)
    st.save(fig, HERE / "figA_law.pdf")


# --------------------------------------------------------------------------------------- Figure 4: sampling
BUDGET = 1e9          # preparations times replayed steps per output step
FIXED = (0, (3.2, 1.4))                 # the full input amplitude
BUDGET_LINES = ((("separate", "gentle_weak"), "-", 0.0), (("separate", "fixed"), FIXED, 0.45),
                (("signed", "gentle_strong"), "-", 0.0), (("signed", "fixed"), FIXED, 0.45),
                (("shared", "fixed"), FIXED, 0.0))


def at_budget(rec, budget):
    """Memory of each reservoir when each output step may use `budget` preparations times replayed steps: three
    settings, each with n = budget / (3 tau_90) preparations that replay the last tau_90 inputs, the range holding 90%
    of the exact recall, from |00000> (run_clock.py budget_replay)."""
    return np.array([m[rec["budgets"].index(budget)] for m in rec["memory"]])


def budget_line(design, rule, budget):
    """Loss rates and memory at a fixed budget along one line, rows over the reservoirs; the gentle lines start from
    the full amplitude at gamma = 1."""
    recs = [r for r in SNAP.get("budget_replay", []) if r["design"] == design
            and (r["rule"] == rule or (rule != "fixed" and r["rule"] == "fixed" and r["gamma"] == 1.0))]
    recs.sort(key=lambda r: r["gamma"])
    return np.array([r["gamma"] for r in recs]), np.array([at_budget(r, budget) for r in recs]).T


def fig_sampling():
    """What survives sampling and why: memory at a fixed budget against the loss rate, and feature variance and
    memory at 10^6 preparations against the write strength at a matched decay-only lifetime, three panels across the
    page; the keys of the line styles sit inside the panels and the key of the colours, which all panels share, in a
    row under them."""
    fig = st.wide_figure(1, hp=88.0, keys=1)
    a, b, c = (st.wide_place(fig, k) for k in range(3))

    # a: memory at a fixed budget of preparations times replayed steps per output step, against the loss rate
    for (design, rule), ls, f in BUDGET_LINES:
        g, rows = budget_line(design, rule, BUDGET)
        col = st.tone(st.FAMILY[design], f)
        st.band(a, g, rows, col, ls=ls, marker="o" if rule != "fixed" else None, ms=st.MS,
                lw=st.LW if rule != "fixed" else st.LW_THIN, alpha=0.15)
    a.set_xscale("log")
    st.decades(a, "x", every=2)
    a.set_xlim(1 / 90, 360)
    a.set_ylim(0, 48)                       # room above the curves for the key
    a.set_yticks([0, 10, 20, 30, 40])
    a.set_xlabel(r"Loss rate $\gamma$")
    a.set_ylabel(r"Capacity $\sum_\tau C_\tau$")

    # b, c: feature variance and memory at 10^6 preparations per setting and step against the write onto the slow
    # modes, every reservoir of every channel that writes at one decay-only lifetime and amplitude; the four channels
    # that carry the story in colour, the others in grey
    P = SNAP["polarization"]
    for fam in sorted(families(), key=lambda f: f in st.MAIN):
        if fam not in P or fam == "dephasing":
            continue
        w = np.array(P[fam]["write_slow"])
        col = st.FAMILY[fam] if fam in st.MAIN else st.OTHER
        for ax, field in ((b, "var_mean"), (c, "shots_1000000")):
            ax.plot(w, np.array(P[fam][field]), "o", ms=st.MS, mfc=col, mec="white", mew=0.5, ls="none")
    # power-law fits within the channels without and with dark states (weak and strong loss at the matched lifetime)
    for group in (("separate", "exchange", "gainloss"), ("pair", "signed", "groups", "shared")):
        w = np.concatenate([P[f]["write_slow"] for f in group])
        v = np.concatenate([P[f]["var_mean"] for f in group])
        k, c0 = np.polyfit(np.log10(w), np.log10(v), 1)
        ww = np.geomspace(w.min() / 1.5, w.max() * 1.5, 10)
        b.plot(ww, 10 ** c0 * ww ** k, color=st.INK3, lw=st.LW_GUIDE, ls=(0, (1, 1.6)), zorder=0)
    # the sum over 300 delays of a readout without any recall, about 300/1600 units
    c.axhline(300 / 1600, color=st.INK3, lw=st.LW_GUIDE, ls=FEATURES, zorder=0)
    for ax in (b, c):
        ax.set_xscale("log")
        ax.set_yscale("log")
        st.decades(ax, "x", labels="power", every=2)
        ax.set_xlim(1.2e-3, 4)
        ax.set_xlabel(r"Write strength")
    st.decades(b, "y", labels="power", every=2)
    b.set_ylim(1e-8, 3e-3)
    b.set_ylabel("Feature variance")
    st.decades(c, "y", labels="plain")
    c.set_ylim(0.1, 60)
    c.set_ylabel(r"Capacity at $10^6$")

    st.fit_row(fig, (a, b, c), (10, 9, 9))
    for ax, name in zip((a, b, c), "abc"):
        st.letter(ax, name)
    st.align(fig, rows=[(a, b, c)])
    st.inkey(a, [(st.line_handle(st.INK3, marker="o"), "gentle writing"),
                 (st.line_handle(st.INK3, ls=FIXED, lw=st.LW_THIN), "full amplitude")], "upper right")
    st.inkey(b, [(st.line_handle(st.INK3, ls=(0, (1, 1.6)), lw=st.LW_GUIDE), "fit")], "lower right")
    st.inkey(c, [(st.line_handle(st.INK3, ls=FEATURES, lw=st.LW_GUIDE), "no recall")], "upper left")
    dot = lambda col: st.marker_handle(col, mfc=col, mec=col, mew=0.5)
    W, _ = st._pts(fig)
    st.key(fig, W / 2, [(dot(st.FAMILY["separate"]), "independent"), (dot(st.FAMILY["shared"]), "equal phases"),
                        (dot(st.FAMILY["signed"]), "opposite phases"), (dot(st.OTHER), "other channels")],
           ncol=4, center=True, handle=5.0, gap=2.5, sep=10.0)
    st.save(fig, HERE / "fig4_sampling.pdf")


# ------------------------------------------------------------------------------- Appendix: other sizes
TAIL = 3 * 20 / 1600      # recall summed over the last twenty delays: three times what a readout without recall gives


def fig_sizes():
    """The memory curves of Fig. 2c at four and six spins: independent decay, equal phases and opposite phases at the
    full amplitude, and gentle writing for independent decay and opposite phases, which starts from the full amplitude
    at gamma = 1. A line stops where its recall reaches past the 300 delays summed. Two panels across one column, each
    with its number of spins above the dashed line at its number of features; the key that the panels share sits
    under them."""
    fig = st.wide_figure(1, width=st.COLUMN, hp=86.0, keys=3)
    axes = [st.wide_place(fig, 0), st.wide_place(fig, 1)]
    for ax, n in zip(axes, ("4", "6")):
        rec = SNAP["sizes"][n]
        lines = {(l["design"], l["rule"]): l for l in rec["lines"]}
        for (fam, rule), line in lines.items():
            x, rows, tail = np.array(line["gamma"]), np.array(line["capacity"]), np.array(line["tail"])
            inside = np.cumsum(tail > TAIL) == 0
            x, rows = x[inside], rows[:, inside]
            if rule == "fixed":
                st.band(ax, x, rows, st.FAMILY[fam], ls=st.DASH[fam])
                continue
            full = lines[(fam, "fixed")]
            one = np.array(full["capacity"])[:rows.shape[0], full["gamma"].index(1.0)]
            x, rows = (np.r_[x, 1.0], np.c_[rows, one]) if fam == "separate" else (np.r_[1.0, x], np.c_[one, rows])
            st.band(ax, x, rows, st.FAMILY[fam], ls=SCALED, marker="o", ms=st.MS, alpha=0.12)
        feats = rec["features"]
        ax.axhline(feats, color=st.INK3, lw=st.LW_GUIDE, ls=FEATURES)
        ax.annotate(rf"$N={n}$", xy=(1 / 30, feats), xycoords="data", xytext=(0, 3.5), textcoords="offset points",
                    ha="left", va="baseline", color=st.INK3, fontsize=st.TXT)
        ax.set_xscale("log")
        st.decades(ax, "x", every=2)
        ax.set_xlim(1 / 40, 400)
        ax.set_ylim(0, 1.25 * feats)        # room above the limit for the number of spins
        ax.set_yticks(np.linspace(0, feats, 4))
        ax.set_xlabel(r"Loss rate $\gamma$")
    axes[0].set_ylabel(r"Capacity $\sum_\tau C_\tau$")
    st.fit_row(fig, axes, (1, 1))
    for ax, name in zip(axes, "ab"):
        st.letter(ax, name)
    st.align(fig, rows=[tuple(axes)])
    shape = lambda f: (st.line_handle(st.FAMILY[f], ls=st.DASH[f]), st.NAME[f])
    W, _ = st._pts(fig)
    st.key(fig, W / 2, [shape("separate"), shape("shared"), shape("signed"),
                        (st.line_handle(st.INK3, ls=SCALED, marker="o"), "gentle writing"),
                        (st.line_handle(st.INK3, ls=FEATURES, lw=st.LW_GUIDE), "number of features")],
           ncol=2, center=True, sep=10.0)
    st.save(fig, HERE / "figE_sizes.pdf")


if __name__ == "__main__":
    fig2()
    fig3()
    fig_sampling()
    fig_darkmodel()
    fig_law()
    fig_sizes()
