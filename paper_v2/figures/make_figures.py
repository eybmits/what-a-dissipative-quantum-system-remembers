#!/usr/bin/env python3
"""Figures 2-5 and the appendix figures (other sizes, supporting results) from
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
SCALED = (0, (3.2, 1.1, 0.9, 1.1))    # scaled amplitude: the input amplitude follows the loss
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
    ax.plot(x, m, color=color, lw=st.LW if lw is None else lw, path_effects=st.halo(lw))


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
    ax.grid(False)
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
    ax.text(0.14, 180, r"\textbf{bright}", color=st.tone(st.FAMILY["shared"], 0.35), fontsize=st.KEY_TXT, va="baseline")
    ax.text(0.14, 7e-4, r"\textbf{dark}", color=st.FAMILY["shared"], fontsize=st.KEY_TXT, va="baseline")


def fig2():
    """The shape sets the direction of the clock: recall at weak and at strong loss, memory against loss and the decay
    rates in one row. The two recall maps share their delay axis and carry the colour scale in an empty corner; the
    memory panel is the widest and carries the key of the shapes above the limit set by the features."""
    fig = st.wide_figure(1, hp=118.0)
    a, b, c, d = (st.wide_place(fig, k) for k in range(4))

    # a, b: recall against loss rate and delay, local loss and one shared channel; dashed: delay below which half of
    # the memory lies
    im = recall_map(a, "separate")
    recall_map(b, "shared")
    b.set_ylabel("")
    b.tick_params(axis="y", which="both", labelleft=False)

    # c: memory against loss for every shape, the four that carry the story in colour and the others in grey;
    # circles: scaled amplitude
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
    c.annotate("45 features", xy=(0.25, 45), xycoords="data", xytext=(0, -8.5),
               textcoords="offset points", ha="left", va="baseline", color=st.INK3, fontsize=st.KEY_TXT)
    c.set_xscale("log")
    st.decades(c, "x", every=2)
    c.set_xlim(1 / 40, 400)
    c.set_xlabel(r"Loss rate $\gamma$")
    c.set_ylim(0, 66)                       # room above the limit for the three rows of the key
    c.set_yticks([0, 15, 30, 45])
    c.set_ylabel(r"Capacity $\sum_\tau C_\tau$")
    st.loss_regions(c, drop=18.5, at=45)

    # d: decay rates, bright and dark modes of the equal-phase channel and the slowest mode of independent decay
    decay_rates(d)

    st.fit_row(fig, (a, b, c, d), (70, 70, 152, 80), joined=(b,))
    for ax, name, title in zip((a, b, c, d), "abcd", ("Independent decay", "Equal phases", "Memory capacity",
                                                      "Bright and dark modes")):
        st.letter(ax, name, title)
    st.align(fig, rows=[(a, b, c, d)])
    st.inset_colour_key(a, im, r"$C_\tau$", [0, 0.5, 1])
    # under the colour scale, in the empty top of a: the white dashed line of both maps
    st.inkey(a, [(st.line_handle("white", ls=(0, (3, 1.5)), under=st.RECALL(0.6)), "median")], "upper right",
             pad=(4.5, 24.0), box=False)
    shape = lambda f: (st.line_handle(st.FAMILY[f], ls=st.DASH[f]), st.NAME[f])
    st.inkey(c, [shape("separate"), shape("signed"),
                 shape("shared"), (st.line_handle(st.OTHER, lw=st.LW_THIN), "other channels"),
                 shape("dephasing"), (st.line_handle(st.INK3, ls=SCALED, marker="o"), "scaled amplitude")],
             "upper left", ncol=2, pad=(3.5, 5.0), handle=8.0, gap=2.5, sep=4.0)
    st.save(fig, HERE / "fig2_clock.pdf")


# ----------------------------------------------------------------------------------------------- Figure 3
def scaling_limit(ax):
    """The scaling limit (lines) against the recursion of the full five-spin reservoir at its weakest leak
    (markers), for one reservoir: the share of single inputs in the capacity (solid, filled) and the variance they
    leave in the features, scaled to the largest value of the limit (dashed, open)."""
    G = SNAP["scaling_limit"]
    for fam in ("separate", "signed"):
        col = st.FAMILY[fam]
        lim = next(r for r in G["limit"] if r["design"] == fam and r["seed"] == G["main"])
        top = max(lim["signal"])
        ax.plot(lim["nu"], lim["share1"], color=col, path_effects=st.halo(), zorder=2)
        ax.plot(lim["nu"], np.array(lim["signal"]) / top, color=col, lw=st.LW_THIN, ls=(0, (3, 1.5)), zorder=2)
        pts = [e for e in G["exact"] if e["design"] == fam and e["seed"] == G["main"]]
        weakest = (max if fam == "signed" else min)(e["gamma"] for e in pts)
        pts = [e for e in pts if e["gamma"] == weakest]
        ax.plot([e["nu"] for e in pts], [e["share1"] for e in pts], "o", ms=st.MS - 0.4, mfc=col, mec=col, mew=0.9,
                ls="none", zorder=4)
        ax.plot([e["nu"] for e in pts], [e["signal"] / top for e in pts], "o", ms=st.MS - 0.4, mfc="white", mec=col,
                mew=0.9, ls="none", zorder=3)
    ax.set_xscale("log")
    st.decades(ax, "x", every=2)
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$\nu=\eta^2/\epsilon$")
    ax.set_ylabel(r"Share, signal")


def fig3():
    """The price of a long memory and how the input pays it: the recall against the memory lifetime, the write along
    and across the phases, the five-spin order against the mixing law and the scaling limit, in one row; the
    keys sit inside the panels (a names its markers, its colours are those of c)."""
    fig = st.wide_figure(1, hp=st.HP_ROW)
    a, b, c, d = (st.wide_place(fig, k) for k in range(4))
    dashed = (0, (3, 1.5))

    # a: linear recall summed over all delays against the memory lifetime, from exact moments: at the fixed amplitude
    # (filled) it peaks and then falls as the memory grows, with scaled amplitude (open) it stays
    for fam in ("separate", "signed"):
        col = st.FAMILY[fam]
        for full in (True, False):
            sel = sorted(five_rows(fam, full), key=lambda r: np.mean(r["t_mem"]))
            x = np.array([np.mean(r["t_mem"]) for r in sel])
            y = np.array([np.mean(r["memory"]) for r in sel])
            a.plot(x, y, color=col, lw=st.LW_THIN, zorder=1)
            a.plot(x, y, "o", ms=st.MS, mfc=col if full else "white", mec=col, mew=0.9, ls="none", zorder=2,
                   clip_on=False)
    a.set_xscale("log")
    st.decades(a, "x", labels="power", every=2)
    a.set_xlim(0.7, 1.4e4)
    a.set_ylim(0, 56)                       # room above the points for the key
    a.set_yticks([0, 15, 30, 45])
    a.set_xlabel(r"Lifetime $t_{\mathrm{mem}}$")
    a.set_ylabel(r"Capacity $\sum_\tau C_\tau$")

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

    # d: the scaling limit against the five-spin recursion
    scaling_limit(d)

    st.fit_row(fig, (a, b, c, d), (80, 76, 118, 78))
    for ax, name, title in zip((a, b, c, d), "abcd", ("Memory against lifetime", "Writing into dark states",
                                                      "Mixing law", "Scaling limit")):
        st.letter(ax, name, title)
    st.align(fig, rows=[(a, b, c, d)])
    st.inkey(a, [(st.marker_handle(st.INK3, mfc=st.INK3, mec=st.INK3, mew=0.9), "fixed amplitude"),
                 (st.marker_handle(st.INK3, mfc="white", mec=st.INK3, mew=0.9), "scaled amplitude")], "upper left")
    st.inkey(b, [(st.line_handle(st.FAMILY["shared"], marker="o"), "matched"),
                 (st.line_handle(st.FAMILY["signed"], marker="o"), "not matched"),
                 (st.line_handle(st.INK3, ls=(0, (1, 1.6)), lw=st.LW_GUIDE), r"$1/\gamma$")], "lower left",
             pad=(3.0, 5.0))
    st.inkey(c, [(st.marker_handle(st.FAMILY["separate"], mfc=st.FAMILY["separate"], mec=st.FAMILY["separate"],
                                   mew=0.9), "independent"),
                 (st.marker_handle(st.FAMILY["signed"], mfc=st.FAMILY["signed"], mec=st.FAMILY["signed"], mew=0.9),
                  "opposite phases"),
                 (st.marker_handle(st.INK3, mfc="white", mec=st.INK3, mew=0.9), "scaled amplitude")], "upper left")
    st.along(c, "mixing law", (1, 1), (10, 10), (2.0, 2.0))
    st.inkey(d, [(st.line_handle(st.INK3, marker="o", mfc=st.INK3), "share"),
                 (st.line_handle(st.INK3, ls=dashed, lw=st.LW_THIN, marker="o"), "signal")], "upper left",
             at=(0.06, 0.66), handle=11.0)
    st.save(fig, HERE / "fig3_price.pdf")


# ------------------------------------------------------------------------------- the dark-state figure
def fig_darkmodel():
    """The dark-state model and the erasure test, two panels across one column that share the recall
    axis, as tall as the panels of Figs. 2 and 3, each with its key in the room above the curves."""
    fig = st.wide_figure(1, width=st.COLUMN, hp=80.0)
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
    recall_axes(b, 300)
    for ax in (a, b):
        ax.set_ylim(0, 1.15)                # the keys sit right of the curves, which end before the longest delays
    b.set_ylabel("")
    b.tick_params(axis="y", which="both", labelleft=False)

    # a is the wider panel: its key is the wider one
    st.fit_row(fig, (a, b), (1, 1), joined=(b,))
    for ax, name, title in zip((a, b), "ab", ("Dark-state model", "Erasure test")):
        st.letter(ax, name, title)
    st.align(fig, rows=[(a, b)])
    st.inkey(a, [(st.line_handle(light), r"$\gamma=4$"), (st.line_handle(st.FAMILY["shared"]), r"$\gamma=64$"),
                 (st.marker_handle(st.INK2, mfc="white", mec=st.INK2, mew=0.9), "model")],
             "upper right", pad=(5.0, 5.5), handle=8.0, gap=2.5)
    st.inkey(b, [(st.line_handle(st.FAMILY["shared"]), "intact"), (st.line_handle(rot), "copy"),
                 (st.line_handle(st.ERASED), "dark")], "upper right", pad=(5.0, 5.5), handle=8.0, gap=2.5)
    st.save(fig, HERE / "figD_dark.pdf")


# ------------------------------------------------------------------------------------- five-spin order and law
def five_rows(fam, full):
    """Entries of the five-spin order runs along the fixed amplitude (full=True) or the line of scaled amplitude, by loss rate."""
    rows = [r for r in SNAP.get("five_orders", []) if r["design"] == fam and "law_mean" in r]
    gentle = lambda g: min(1.0, g ** (-0.5 if fam == "signed" else 0.5))
    sel = [r for r in rows if (r["eta"] == 1.0) == full and (full or abs(r["eta"] - gentle(r["gamma"])) < 1e-12)]
    return sorted(sel, key=lambda r: r["gamma"])


def five_law(ax):
    """Mean order of the five-spin memory against the mixing law's s t_mem: fixed amplitude (filled, along the loss
    rate) and scaled amplitude (open); dashed: the law."""
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
    longest memory of each shape at the fixed amplitude, and the exponential law."""
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


# --------------------------------------------------------------------------------------- Figure 4: sampling
BUDGET = 1e9          # shots times replayed steps per output step
FIXED = (0, (3.2, 1.4))                 # the full input amplitude
STRONG_LOSS = (0, (1.0, 1.4))           # independent decay at the strong loss rate of the collective channels
BUDGET_LINES = ((("separate", "gentle_weak"), "-", 0.0), (("separate", "fixed"), FIXED, 0.45),
                (("signed", "gentle_strong"), "-", 0.0), (("signed", "fixed"), FIXED, 0.45),
                (("shared", "fixed"), FIXED, 0.0))


def at_budget(rec, budget):
    """Memory of each reservoir when each output step may use `budget` shots times replayed steps: three
    settings, each with n = budget / (3 tau_90) shots that replay the last tau_90 inputs, the range holding 90%
    of the exact recall, from |00000> (run_clock.py budget_replay)."""
    return np.array([m[rec["budgets"].index(budget)] for m in rec["memory"]])


def budget_line(design, rule, budget):
    """Loss rates and memory at a fixed budget along one line, rows over the reservoirs; the line of scaled amplitudes start from
    the fixed amplitude at gamma = 1."""
    recs = [r for r in SNAP.get("budget_replay", []) if r["design"] == design
            and (r["rule"] == rule or (rule != "fixed" and r["rule"] == "fixed" and r["gamma"] == 1.0))]
    recs.sort(key=lambda r: r["gamma"])
    return np.array([r["gamma"] for r in recs]), np.array([at_budget(r, budget) for r in recs]).T


REPLAY_LINES = ((("separate", "gentle_weak", 0.03125), "-", "o"), (("separate", "fixed", 0.03125), FIXED, None),
                (("signed", "gentle_strong", 64.0), "-", "o"), (("signed", "fixed", 64.0), FIXED, None),
                (("shared", "fixed", 64.0), FIXED, None), (("separate", "fixed", 64.0), STRONG_LOSS, None))


def fig_sampling():
    """What survives sampling and why: memory against the number of shots, and feature variance and the
    recovered fraction of the capacity against the stationary response at a matched relaxation time, three panels
    across the page; the key of the line styles sits in a and the key of the colours, which all panels share, in b."""
    fig = st.wide_figure(1, hp=st.HP_ROW)
    a, b, c = (st.wide_place(fig, k) for k in range(3))

    # a: memory against the shots per setting and step, features from the replay of the last tau_90 inputs:
    # independent decay at weak loss (gamma = 1/32) and, at strong loss (gamma = 64), the two collective channels and
    # independent decay
    lines = {(r["design"], r["rule"], r["gamma"]): r for r in SNAP["replay_shots"]["uniform"]}
    for key, ls, marker in REPLAY_LINES:
        rec = lines[key]
        st.band(a, np.array(rec["shots"], float), np.array(rec["memory"]), st.FAMILY[key[0]], ls=ls, marker=marker,
                ms=st.MS, lw=st.LW if marker else st.LW_THIN, alpha=0.15)
    a.set_xscale("log")
    st.decades(a, "x", labels="power", every=2)
    a.set_xlim(7e2, 5e8)
    a.set_ylim(0, 47)                       # the key sits above the rising curves, in the upper left
    a.set_yticks([0, 10, 20, 30])
    a.set_xlabel(r"Shots $n$")
    a.set_ylabel(r"Capacity $\sum_\tau C_\tau$")

    # b, c: feature variance, and the fraction of the exact capacity that 10^6 shots per setting and step
    # recover, against the write onto the slow modes: every reservoir of every channel that writes, at one decay-only
    # lifetime and amplitude; the four channels that carry the story in colour, the others in grey
    P = SNAP["polarization"]
    for fam in sorted(families(), key=lambda f: f in st.MAIN):
        if fam not in P or fam == "dephasing":
            continue
        w = np.array(P[fam]["write_slow"])
        col = st.FAMILY[fam] if fam in st.MAIN else st.OTHER
        b.plot(w, np.array(P[fam]["var_mean"]), "o", ms=st.MS, mfc=col, mec="white", mew=0.5, ls="none")
        c.plot(w, np.array(P[fam]["shots_1000000"]) / np.array(P[fam]["exact"]), "o", ms=st.MS, mfc=col, mec="white",
               mew=0.5, ls="none")
    for ax in (b, c):
        ax.set_xscale("log")
        ax.set_yscale("log")
        st.decades(ax, "x", labels="power", every=2)
        ax.set_xlim(1.2e-3, 4)
        ax.set_xlabel(r"Stationary response")
    st.decades(b, "y", labels="power", every=2)
    b.set_ylim(1e-8, 3e-3)
    b.set_ylabel("Feature variance")
    st.decades(c, "y", labels="plain")
    c.set_ylim(2e-3, 2)
    c.set_ylabel("Recovered fraction")
    from scipy import stats
    shown = [f for f in families() if f in P and f != "dephasing"]
    rho = stats.spearmanr(np.concatenate([P[f]["write_slow"] for f in shown]),
                          np.concatenate([np.array(P[f]["shots_1000000"]) / np.array(P[f]["exact"]) for f in shown]))[0]
    c.annotate(rf"rank correlation {rho:.2f}", xy=(1, 0), xycoords="axes fraction", xytext=(-6, 7),
               textcoords="offset points", ha="right", va="baseline", color=st.INK3, fontsize=st.KEY_TXT)

    st.fit_row(fig, (a, b, c), (10, 9, 9))
    for ax, name, title in zip((a, b, c), "abc", ("Finite sampling", "Signal size", "Recovered capacity")):
        st.letter(ax, name, title)
    st.align(fig, rows=[(a, b, c)])
    st.inkey(a, [(st.line_handle(st.INK3, marker="o"), "scaled amplitude"),
                 (st.line_handle(st.INK3, ls=FIXED, lw=st.LW_THIN), "fixed amplitude"),
                 (st.line_handle(st.FAMILY["separate"], ls=STRONG_LOSS, lw=st.LW_THIN), r"independent, $\gamma=64$")],
             "upper left")
    dot = lambda col: st.marker_handle(col, mfc=col, mec=col, mew=0.5)
    st.inkey(b, [(dot(st.FAMILY["separate"]), "independent"), (dot(st.FAMILY["shared"]), "equal phases"),
                 (dot(st.FAMILY["signed"]), "opposite phases"), (dot(st.OTHER), "other channels")], "upper left",
             handle=5.0, gap=2.5)
    st.save(fig, HERE / "fig4_sampling.pdf")


def fig_checks():
    """Appendix E: two supporting results across one column. a: the collapse of the single-input recall time, eta^2 t_f
    against gamma/eta^2 (independent decay) and gamma eta^2 (opposite phases), one marker per amplitude. b: memory at a
    fixed budget of shots times replayed steps per output step against the loss rate. The key of the line
    styles of b sits above its curves; the keys of the colours and of the markers of a sit in rows under the panels."""
    fig = st.wide_figure(1, width=st.COLUMN, hp=112.0)
    a, b = st.wide_place(fig, 0), st.wide_place(fig, 1)
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
    a.set_ylim(1e-3, 1e7)                   # room above the points for the keys
    a.set_yticks([1e-2, 1e0, 1e2])
    a.set_xlabel(r"Loss $\gamma/\eta^2$ or $\gamma\eta^2$")
    a.set_ylabel(r"Recall time $\eta^2\,t_{\mathrm f}$")


    for (design, rule), ls, f in BUDGET_LINES:
        g, rows = budget_line(design, rule, BUDGET)
        col = st.tone(st.FAMILY[design], f)
        st.band(b, g, rows, col, ls=ls, marker="o" if rule != "fixed" else None, ms=st.MS,
                lw=st.LW if rule != "fixed" else st.LW_THIN, alpha=0.15)
    b.set_xscale("log")
    st.decades(b, "x", every=2)
    b.set_xlim(1 / 90, 360)
    b.set_ylim(0, 60)                       # room above the curves for the key
    b.set_yticks([0, 10, 20, 30, 40])
    b.set_xlabel(r"Loss rate $\gamma$")
    b.set_ylabel(r"Capacity $\sum_\tau C_\tau$")
    st.loss_regions(b, drop=0.0, at=36.5, labels=("weak", "strong"))

    st.fit_row(fig, (a, b), (1, 1))
    for ax, name, title in zip((a, b), "ab", ("Recall-time collapse", "Fixed budget")):
        st.letter(ax, name, title)
    st.align(fig, rows=[(a, b)])
    dot = lambda f: (st.marker_handle(st.FAMILY[f], mfc=st.FAMILY[f], mec=st.FAMILY[f], mew=0.5), st.NAME[f])
    st.inkey(a, [dot("separate"), dot("signed")], "upper left", handle=5.0, gap=2.5, box=False)
    st.inkey(a, [(st.marker_handle(st.INK2, marker=m, ms=2.9, mfc="white", mec=st.INK2, mew=0.8), label)
                 for m, label in zip(marks, (r"$\eta=1$", r"$1/2$", r"$1/4$", r"$1/8$"))],
             "upper left", ncol=2, pad=(6.0, 6.5 + 2 * st.KEY_ROW), handle=5.0, gap=2.5, sep=8.0, box=False)
    st.frame(fig, 6)
    st.inkey(b, [(st.line_handle(st.INK3, marker="o"), "scaled amplitude"),
                 (st.line_handle(st.INK3, ls=FIXED, lw=st.LW_THIN), "fixed amplitude"), dot("shared")], "upper right")
    st.save(fig, HERE / "figE_checks.pdf")


# ------------------------------------------------------------------------------- Appendix: other sizes
TAIL = 3 * 20 / 1600      # recall summed over the last twenty delays: three times what a readout without recall gives


def fig_sizes():
    """The memory curves of Fig. 2c at four and six spins: independent decay, equal phases and opposite phases at the
    fixed amplitude, and scaled amplitude for independent decay and opposite phases, which starts from the fixed amplitude
    at gamma = 1. A line stops where its recall reaches past the 300 delays summed. Two panels across one column, each
    with its number of spins above the dashed line at its number of features; the key that the panels share sits
    under them."""
    fig = st.wide_figure(1, width=st.COLUMN, hp=112.0)
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
        ax.set_xscale("log")
        st.decades(ax, "x", every=2)
        ax.set_xlim(1 / 40, 400)
        ax.set_ylim(0, 1.47 * feats)        # room above the limit for the keys
        ax.set_yticks(np.linspace(0, feats, 4))
        ax.set_xlabel(r"Loss rate $\gamma$")
        st.loss_regions(ax, drop=9.5, at=feats, labels=("weak", "strong"))
    axes[0].set_ylabel(r"Capacity $\sum_\tau C_\tau$")
    st.fit_row(fig, axes, (1, 1))
    for ax, name, title in zip(axes, "ab", ("Four spins", "Six spins")):
        st.letter(ax, name, title)
    st.align(fig, rows=[tuple(axes)])
    shape = lambda f: (st.line_handle(st.FAMILY[f], ls=st.DASH[f]), st.NAME[f])
    st.inkey(axes[0], [shape("separate"), shape("shared"), shape("signed")], "upper left")
    st.inkey(axes[1], [(st.line_handle(st.INK3, ls=SCALED, marker="o"), "scaled amplitude"),
                       (st.line_handle(st.INK3, ls=FEATURES, lw=st.LW_GUIDE), "all features")], "upper left")
    st.save(fig, HERE / "figE_sizes.pdf")


if __name__ == "__main__":
    fig2()
    fig3()
    fig_sampling()
    fig_darkmodel()
    fig_sizes()
    fig_checks()
