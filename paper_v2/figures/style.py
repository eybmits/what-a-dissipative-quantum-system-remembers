"""Figure style of the paper.

Figures across the page hold one row of four panels (fit_row): each gutter is as wide as the tick and axis labels of
its panel need, panels that share an axis sit side by side, and the panels share the rest of the width, so that they
are as large as the page allows; keys and colour scales sit inside the panels, in space the data leave free (inkey,
inset_colour_key). The one-column figures of the appendix keep the older grid, in TeX points (Quantum class, 17 cm
text width, two columns of 231.9 pt with 20 pt between them): each text column holds two panels of about 71 x 66 pt,
each with a 40 pt gutter on its left, and their keys sit in a band under the panels. Panels carry no titles: a bold
letter sits at the left edge of each panel's column, in a row above the axes, and x labels share one baseline within
a row. All text is typeset by LaTeX in Latin Modern Roman at 10 pt, the size of the captions and the font of the
text, with the mathematics of the text. One colour stands for one
shape of the environment throughout: warm colours for shapes without a protected subspace, cool colours for shapes
with one. Four shapes carry the story; where all eight share a panel, the other four are drawn in grey.
"""
from __future__ import annotations

import os
import shutil

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

if shutil.which("latex") is None:
    os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

PT = 1 / 72.27
PAGE = 170.0 / 25.4                 # text width of the Quantum class, two columns (17 cm)
COLUMN = (PAGE - 20 * PT) / 2       # column width (column separation 20 pt)

# vertical layout, in points: above the axes (the row of panel letters, or only the top tick label of a figure
# without letters), between panel rows, x labels
TOP, TOP_PLAIN, GAP, BOTTOM = 13.0, 8.0, 44.0, 29.0
RISE = 5.0                          # baseline of the panel letters above the axes
# keys sit in a band under the panels: first baseline below the top of the band, distance between baselines, length
# of a handle, space between handle and text, space between columns
KEY_TOP, KEY_ROW, HANDLE, HANDLE_PAD, KEY_SEP = 10.0, 12.0, 11.0, 3.5, 7.0

# the same colours as figures/fig1_concept.tex
INK, INK2, INK3, RULE, SHADE = "#1a1a1a", "#3a3a3a", "#6f6f6f", "#d9d9d9", "#f2f2f2"
FAMILY = {
    "separate": "#c8553d",    # local loss
    "exchange": "#c9852a",    # local loss with exchange
    "gainloss": "#8f2d56",    # local gain and loss
    "dephasing": "#6b7385",   # local dephasing
    "pair": "#2a9d8f",        # pair loss
    "signed": "#7b5ea7",      # one shared channel, opposite phases
    "groups": "#4f86c6",      # two shared channels
    "shared": "#16325c",      # one shared channel, equal phases
}
NAME = {
    "separate": "local", "exchange": "exchange", "gainloss": "gain/loss", "dephasing": "dephasing",
    "pair": "pairwise", "signed": "opposite phases", "groups": "two channels", "shared": "shared",
}
STRONG, GENTLE, ERASED = "#9c6644", "#2d6a4f", "#8c8c8c"
# one set of sizes for every figure: data lines, lines in the background, guide lines, markers, text
LW, LW_THIN, LW_GUIDE, MS, TXT = 1.2, 0.9, 0.7, 3.2, 10.0
# the four shapes that carry the story, and the grey of the others; shapes without protected operators dashed,
# dephasing dotted, shapes with dark states solid
MAIN = ("separate", "dephasing", "shared", "signed")
OTHER = "#b4b4b4"
DASH = {"separate": (0, (4, 1.6)), "exchange": (0, (4, 1.6)), "gainloss": (0, (4, 1.6)),
        "dephasing": (0, (1.2, 1.4)), "pair": "-", "signed": "-", "groups": "-", "shared": "-"}

# white for no memory, dark for full memory
RECALL = mcolors.LinearSegmentedColormap.from_list(
    "recall", ["#ffffff", "#fbe7c6", "#f4a259", "#c8553d", "#6b2242", "#21102b"])

mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}"
                           # matplotlib drops glyph 0 of the TeX symbol font (the minus) when it
                           # subsets Type 1 fonts; take the minus from the text font instead
                           r"\DeclareMathSymbol{-}{\mathbin}{operators}{123}",
    "font.family": "serif",
    "font.size": TXT, "axes.labelsize": TXT, "axes.titlesize": TXT,
    "xtick.labelsize": TXT, "ytick.labelsize": TXT, "legend.fontsize": TXT, "legend.title_fontsize": TXT,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "axes.grid": False, "axes.axisbelow": True,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
    "xtick.major.pad": 3.0, "ytick.major.pad": 3.0, "axes.labelpad": 3.0,
    "lines.linewidth": LW, "lines.markersize": MS, "lines.markeredgewidth": 0.8, "patch.linewidth": 0.6,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "lines.dash_capstyle": "butt",
    "legend.frameon": False, "legend.borderpad": 0.2, "legend.borderaxespad": 0.6, "legend.handlelength": 1.8,
    "legend.handletextpad": 0.5, "legend.labelspacing": 0.25, "legend.columnspacing": 1.2,
    "axes.unicode_minus": False, "pdf.compression": 9, "savefig.dpi": 300,
})


def tone(c, f):
    """Lighter (f > 0) or darker (f < 0) variant of a colour."""
    rgb = np.array(mcolors.to_rgb(c))
    return mcolors.to_hex(rgb + (1 - rgb) * f if f >= 0 else rgb * (1 + f))


def align(fig, columns=(), rows=()):
    """Put the y labels of each group of axes (a column) and the x labels of each group (a row) at one distance
    from the axes: the distance the widest or deepest tick labels of the group need, plus the label pad."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = mpl.rcParams["axes.labelpad"] * fig.dpi / 72
    for group in columns:
        need = 0.0
        for ax in group:
            box = ax.get_window_extent(r)
            ext = [t.get_window_extent(r) for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
            need = max([need] + [box.x0 - e.x0 for e in ext])
        for ax in group:
            ax.yaxis.set_label_coords(-(need + pad) / ax.get_window_extent(r).width, 0.5)
    for group in rows:
        need = 0.0
        for ax in group:
            box = ax.get_window_extent(r)
            ext = [t.get_window_extent(r) for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
            need = max([need] + [box.y0 - e.y0 for e in ext])
        # x labels share one baseline, whatever superscripts or roots they carry
        rise = 0.72 * mpl.rcParams["axes.labelsize"] * fig.dpi / 72
        for ax in group:
            ax.xaxis.label.set_verticalalignment("baseline")
            ax.xaxis.set_label_coords(0.5, -(need + pad + rise) / ax.get_window_extent(r).height)


# --------------------------------------------------------------------------------------------- the panel grid
# Every data panel has the same size. Each text column holds two panels, each with a gutter on its left for
# its tick and axis labels, so a figure across the page holds four, aligned with the two text columns; a panel may
# span both grid columns of a text column.
G = 40.0                                    # gutter left of every panel
RIGHT = 10.0                                # margin right of the second panel of a text column, for its last tick label
SEP = 20.0                                  # separation of the two text columns
COLUMN_PT = COLUMN / PT
WP = (COLUMN_PT - 2 * G - RIGHT) / 2        # panel width, about 71 pt
HP = 66.0                                   # panel height
HP_ROW = 100.0                              # panel height of a row of four panels across the page (fit_row)


def grid_x0(col):
    """Left spine of grid column `col`: 0 and 1 in the first text column, 2 and 3 in the second."""
    half, k = divmod(col, 2)
    return half * (COLUMN_PT + SEP) + G + k * (WP + G)


def wide_figure(rows=1, keys=0, width=PAGE, letters=True, hp=None):
    """A figure on the panel grid, across the page or (width=COLUMN) across one text column, with the given number
    of panel rows (of height `hp`, by default HP) and a band of `keys` rows of keys under them."""
    hp = HP if hp is None else hp
    top = TOP if letters else TOP_PLAIN
    below = KEY_TOP + (keys - 1) * KEY_ROW + 4.0 if keys else 0.0
    height = top + rows * hp + (rows - 1) * GAP + BOTTOM + below
    fig = plt.figure(figsize=(width, height * PT))
    fig.grid_top = height - top
    fig.hp = hp
    fig.below = below
    fig.key_boxes = []                      # (label, (x0, y0, x1, y1) in points) of every key entry
    return fig


def wide_place(fig, col, span=1, row=0):
    """Axes of the panel that starts in grid column `col` and spans `span` columns of the same text column."""
    W, Hf = fig.get_size_inches() / PT
    x0 = grid_x0(col)
    x1 = grid_x0(col + span - 1) + WP
    hp = getattr(fig, "hp", HP)
    y1 = fig.grid_top - row * (hp + GAP)
    return fig.add_axes([x0 / W, (y1 - hp) / Hf, (x1 - x0) / W, hp / Hf])


def letter(ax, name, col=None):
    """Bold letter at the left edge of the panel's column (the grid column `col`, or the column fit_row gave the
    panel), in a row above the axes."""
    fig = ax.figure
    W, Hf = fig.get_size_inches() / PT
    y = ax.get_position().y1 * Hf + RISE
    x = ax.gutter_x if col is None else grid_x0(col) - G
    fig.text(x / W, y / Hf, rf"\textbf{{{name}}}", ha="left", va="baseline", fontsize=TXT)


def _shown_labels(axis):
    """Labels of the major ticks inside the view limits (matplotlib also labels the ticks beyond them, but does not
    draw those)."""
    lo, hi = sorted(axis.get_view_interval())
    tol = 1e-9 * max(abs(lo), abs(hi))
    locs = axis.get_majorticklocs()
    return [t.label1 for t, x in zip(axis.get_major_ticks(len(locs)), locs) if lo - tol <= x <= hi + tol]


def fit_row(fig, axes, weights, joined=(), margin=2.0, gap=6.0):
    """Lay one row of panels across the figure: each gutter holds the tick and axis labels of its panel (and what the
    panel before it lets stick out on the right) plus `margin` points; a panel in `joined` shares the y axis of the
    panel before it and sits `gap` points from it; the panels share the rest of the width in proportion to
    `weights`. Sets ax.gutter_x, the left edge of each panel's column, for its letter."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, Hf = _pts(fig)
    ppt = fig.dpi * PT
    left, right = [], []
    for ax in axes:
        box = ax.get_window_extent(r)
        parts = [ax.xaxis.label, ax.yaxis.label] + _shown_labels(ax.xaxis) + _shown_labels(ax.yaxis)
        ext = [t.get_window_extent(r) for t in parts if t.get_visible() and t.get_text()]
        left.append(max([0.0] + [(box.x0 - e.x0) / ppt for e in ext]))
        right.append(max([0.0] + [(e.x1 - box.x1) / ppt for e in ext]))
    gutters = [gap if ax in joined else (right[i - 1] + left[i] + margin if i else left[i] + 0.5)
               for i, ax in enumerate(axes)]
    avail = W - sum(gutters) - right[-1] - 0.5
    x = 0.0
    for i, ax in enumerate(axes):
        x += gutters[i]
        p = ax.get_position()
        ax.set_position([x / W, p.y0, avail * weights[i] / sum(weights) / W, p.height])
        ax.gutter_x = x if ax in joined else x - left[i] - 0.5
        x += avail * weights[i] / sum(weights)


def column_figure(keys=0, height=None):
    """A one-column figure with one panel across both grid columns, without a letter; `height` replaces the panel
    height."""
    fig = wide_figure(1, keys=keys, width=COLUMN, letters=False, hp=height)
    return fig, wide_place(fig, 0, span=2)


# ------------------------------------------------------------------------------------------------------- keys
# A key is drawn by hand in the band under the panels, so that every row has the same height and every handle sits
# at the middle of the lower-case letters: entries are (handle, label), a handle draws itself between two x positions
# at one height, all in points.
MID = 2.6                                   # height of a handle above the baseline of its label


def _pts(fig):
    return fig.get_size_inches() / PT


def _line(fig, xs, ys, **kw):
    W, Hf = _pts(fig)
    fig.add_artist(Line2D(np.asarray(xs) / W, np.asarray(ys) / Hf, transform=fig.transFigure, **kw))


def line_handle(color, ls="-", lw=None, marker=None, ms=None, mfc="white", band=None):
    """A line, with a marker in its middle and with a light band behind it (alpha `band`) if asked."""
    def draw(fig, x0, x1, y):
        if band:
            W, Hf = _pts(fig)
            fig.add_artist(Rectangle((x0 / W, (y - 3.0) / Hf), (x1 - x0) / W, 6.0 / Hf, transform=fig.transFigure,
                                     color=color, alpha=band, lw=0))
        _line(fig, [x0, x1], [y, y], color=color, ls=ls, lw=LW if lw is None else lw, solid_capstyle="butt",
              dash_capstyle="butt")
        if marker:
            _line(fig, [(x0 + x1) / 2], [y], ls="none", marker=marker, ms=MS if ms is None else ms, mfc=mfc,
                  mec=color, color=color)
    return draw


def marker_handle(colors, marker="o", ms=None, mfc=None, mec="white", mew=0.5):
    """One marker, or several side by side (for one shape at several loss rates)."""
    colors = [colors] if isinstance(colors, str) else list(colors)
    def draw(fig, x0, x1, y):
        xs = np.linspace(x0, x1, len(colors) + 2)[1:-1] if len(colors) > 1 else [(x0 + x1) / 2]
        for x, c in zip(xs, colors):
            _line(fig, [x], [y], ls="none", marker=marker, ms=MS if ms is None else ms, mfc=c if mfc is None else mfc,
                  mec=c if mec is None else mec, mew=mew)
    return draw


def patch_handle(color, alpha=1.0):
    def draw(fig, x0, x1, y):
        W, Hf = _pts(fig)
        fig.add_artist(Rectangle((x0 / W, (y - 3.0) / Hf), (x1 - x0) / W, 6.0 / Hf, transform=fig.transFigure,
                                 color=color, alpha=alpha, lw=0))
    return draw


def key_baseline(fig, row):
    return fig.below - KEY_TOP - row * KEY_ROW


ASC, DESC = 7.0, 2.0                        # height of a 10 pt label above and below its baseline


def key(fig, x, entries, ncol=1, row=0, right=False, y=None):
    """Entries in rows of `ncol` columns, the first handle at x points from the left edge of the figure (or, with
    right=True, the end of the longest label of the last column at x); each column as wide as its longest label.
    The first row sits on the baseline y (points from the bottom of the figure), by default in the band under the
    panels."""
    W, Hf = _pts(fig)
    base = (lambda r: key_baseline(fig, r)) if y is None else (lambda r: y - (r - row) * KEY_ROW)
    texts = []
    for i, (handle, label) in enumerate(entries):
        r = row + i // ncol
        texts.append(fig.text(0, base(r) / Hf, label, ha="left", va="baseline", fontsize=TXT))
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    widths = [t.get_window_extent(rend).width / fig.dpi / PT for t in texts]
    colw = [max(widths[c::ncol]) for c in range(ncol)]
    if right:
        x -= sum(colw) + ncol * (HANDLE + HANDLE_PAD) + (ncol - 1) * KEY_SEP
    x0 = [x + sum(colw[:c]) + c * (HANDLE + HANDLE_PAD + KEY_SEP) for c in range(ncol)]
    for i, ((handle, label), t) in enumerate(zip(entries, texts)):
        c, r = i % ncol, row + i // ncol
        yb = base(r)
        handle(fig, x0[c], x0[c] + HANDLE, yb + MID)
        t.set_x((x0[c] + HANDLE + HANDLE_PAD) / W)
        fig.key_boxes.append((label, (x0[c], yb - DESC, x0[c] + HANDLE + HANDLE_PAD + widths[i], yb + ASC)))
    return x0[-1] + HANDLE + HANDLE_PAD + colw[-1]


def inkey(ax, entries, loc="upper left", ncol=1, pad=(4.5, 5.0), at=None):
    """A key inside a panel, in the corner `loc` ('upper left', 'upper right', 'lower left' or 'lower right'), `pad`
    points from the frame (across, up or down), or with that corner at `at` (axes fraction)."""
    fig = ax.figure
    W, Hf = _pts(fig)
    p = ax.get_position()
    vert, horiz = loc.split()
    rows = -(-len(entries) // ncol)
    if at is None:
        cx = p.x0 * W + pad[0] if horiz == "left" else p.x1 * W - pad[0]
        cy = p.y1 * Hf - pad[1] if vert == "upper" else p.y0 * Hf + pad[1]
    else:
        cx, cy = (p.x0 + at[0] * p.width) * W, (p.y0 + at[1] * p.height) * Hf
    y = cy - ASC if vert == "upper" else cy + DESC + (rows - 1) * KEY_ROW
    return key(fig, cx, entries, ncol=ncol, y=y, right=horiz == "right")


def colour_key(fig, im, x, x1, title, ticks, row=0):
    """Horizontal colour scale in the key band: the title at x, the bar from after the title to x1 (points), tick
    labels under the bar."""
    W, Hf = _pts(fig)
    y = key_baseline(fig, row)
    t = fig.text(x / W, y / Hf, title, ha="left", va="baseline", fontsize=TXT)
    fig.canvas.draw()
    x0 = x + t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi / PT + 6.0
    cax = fig.add_axes([x0 / W, (y + MID - 2.6) / Hf, (x1 - x0) / W, 5.2 / Hf])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_ticks(ticks)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    cbar.outline.set_linewidth(0.6)
    cax.tick_params(length=2.0, width=0.6, direction="out", pad=1.6, top=False, labelsize=TXT, color=INK)
    return cax


def inset_colour_key(ax, im, title, ticks, length=36.0, pad=(4.5, 5.0), stacked=False):
    """Horizontal colour scale in the upper right corner of a map, where the map is empty: the title left of the bar
    (or, stacked, above it, the bar centred under the title), tick labels under the bar."""
    fig = ax.figure
    W, Hf = _pts(fig)
    p = ax.get_position()
    y = p.y1 * Hf - pad[1] - ASC                       # baseline of the title; the bar centred on its lower case
    bx1 = p.x1 * W - pad[0] - 3.0                       # room for half the last tick label
    bx0 = bx1 - length
    t = fig.text(0, y / Hf, title, ha="left", va="baseline", fontsize=TXT)
    fig.canvas.draw()
    tw = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi / PT
    if stacked:
        tx0 = p.x1 * W - pad[0] - tw
        t.set_x(tx0 / W)
        bx0 = tx0 + (tw - length) / 2
        by = y - DESC - 2.5 - 4.6                       # bottom of the bar, under the title
        cax = fig.add_axes([bx0 / W, by / Hf, length / W, 4.6 / Hf])
        cax.set_gid("key")
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_ticks(ticks)
        cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        cbar.outline.set_linewidth(0.6)
        cax.tick_params(length=2.0, width=0.6, direction="out", pad=1.6, top=False, labelsize=TXT, color=INK)
        fig.key_boxes.append((title, (min(tx0, bx0 - 7.0), by - 11.3, p.x1 * W - pad[0], y + ASC)))
        return cax
    t.set_x((bx0 - 4.0 - tw) / W)
    cax = fig.add_axes([bx0 / W, (y + MID - 2.3) / Hf, length / W, 4.6 / Hf])
    cax.set_gid("key")
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_ticks(ticks)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    cbar.outline.set_linewidth(0.6)
    cax.tick_params(length=2.0, width=0.6, direction="out", pad=1.6, top=False, labelsize=TXT, color=INK)
    fig.key_boxes.append((title, (bx0 - 4.0 - tw, y - 11.0, bx1 + 3.0, y + ASC)))
    return cax


# ------------------------------------------------------------------------------------------------------ ticks
def decades(ax, axis="x", labels="plain", every=1):
    """Decade ticks with unlabelled minor ticks; labels as plain numbers or as powers of ten, on every
    `every`-th decade."""
    def fmt(v, _):
        e = int(round(np.log10(v)))
        if e % every:
            return ""
        return f"{v:g}" if labels == "plain" else rf"$10^{{{e}}}$"
    for a in ([ax.xaxis, ax.yaxis] if axis == "both" else [getattr(ax, axis + "axis")]):
        a.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
        a.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
        a.set_major_formatter(ticker.FuncFormatter(fmt))
        a.set_minor_formatter(ticker.NullFormatter())


def log_on_linear(ax, axis, lo, hi, labels, every=1):
    """Decade ticks for an axis that shows log10 of a quantity on a linear scale (heat maps); labels on every
    `every`-th decade."""
    a = getattr(ax, axis + "axis")
    majors = [e for e in range(int(np.floor(lo)), int(np.ceil(hi)) + 1) if lo <= e <= hi]
    a.set_major_locator(ticker.FixedLocator(majors))
    a.set_major_formatter(ticker.FixedFormatter(
        ["" if e % every else (f"{10.0 ** e:g}" if labels == "plain" else rf"$10^{{{e}}}$") for e in majors]))
    minors = [e + np.log10(k) for e in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 1) for k in range(2, 10)]
    a.set_minor_locator(ticker.FixedLocator([m for m in minors if lo <= m <= hi]))
    a.set_minor_formatter(ticker.NullFormatter())


# ------------------------------------------------------------------------------------------------------- data
def band(ax, x, rows, color, label=None, lw=None, alpha=0.15, marker=None, ls="-", **kw):
    """Mean over reservoirs with the range between the 16th and 84th percentiles."""
    rows = np.asarray(rows, float)
    mean = np.nanmean(rows, axis=0)
    lo, hi = np.nanpercentile(rows, 16, axis=0), np.nanpercentile(rows, 84, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)
    # markers stay whole at the frame; the data of a marked line lie inside the axes
    ax.plot(x, mean, color=color, lw=LW if lw is None else lw, label=label, marker=marker, mfc="white", mec=color,
            ls=ls, clip_on=marker is None, **kw)
    return mean


def save(fig, path):
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
