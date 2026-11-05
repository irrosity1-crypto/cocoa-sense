"""Visual layer: a dark trading-terminal look shared by every page.

* ``THEME`` holds the palette (panels, text, accent, up/down colours).
* ``apply_style`` injects the page CSS on each run; the Plotly template is registered on import.
* ``ticker`` renders the scrolling price strip shown under every page title.
* ``cards`` renders a row of price/KPI cards with coloured up/down badges.

Pure presentation: nothing here reads or changes model results.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

THEME = {
    "bg": "#0b0e13",
    "panel": "#11151c",
    "panel2": "#161b24",
    "border": "#232a36",
    "grid": "#1d2430",
    "text": "#e6e9ef",
    "muted": "#8a93a3",
    "accent": "#e8a33d",
    "up": "#16c784",
    "down": "#ea3943",
    "info": "#4c8dff",
}
ACTUAL = THEME["text"]  # line colour for actual prices in every chart


def _register_template() -> None:
    t = THEME
    axis = {"gridcolor": t["grid"], "linecolor": t["border"], "zerolinecolor": t["border"]}
    template = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": "#c9d1dc", "size": 12},
            colorway=["#4c8dff", "#2dd4bf", "#a78bfa", "#fbbf24", "#9aa4b2", "#f472b6"],
            xaxis=axis,
            yaxis=axis,
            legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#c9d1dc"}},
            hoverlabel={
                "bgcolor": t["panel2"],
                "bordercolor": t["border"],
                "font": {"color": t["text"]},
            },
            title={"font": {"color": t["text"], "size": 14}},
            margin={"t": 40, "b": 30, "l": 10, "r": 10},
        )
    )
    pio.templates["cocoasense"] = template
    pio.templates.default = "plotly_dark+cocoasense"


_register_template()

CSS = (
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
  --bg: %(bg)s; --panel: %(panel)s; --panel2: %(panel2)s; --border: %(border)s;
  --text: %(text)s; --muted: %(muted)s; --accent: %(accent)s; --up: %(up)s; --down: %(down)s;
  --mono: "JetBrains Mono", ui-monospace, Consolas, monospace;
  --sans: Inter, "Segoe UI", Arial, sans-serif;
}
html, body, [data-testid="stAppViewContainer"] { font-family: var(--sans); }
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 500px at 70%% -10%%, rgba(232,163,61,0.08), transparent 60%%), var(--bg);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1400px; }
h1 { font-weight: 800 !important; letter-spacing: -0.02em; font-size: 2rem !important; }
h2, h3 { font-weight: 700 !important; letter-spacing: -0.01em; }
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid var(--border); }
[data-testid="stSidebarNav"] a { border-radius: 8px; }
[data-testid="stMetric"] {
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
}
[data-testid="stMetricValue"] { font-family: var(--mono); font-weight: 600; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; text-transform: uppercase;
  font-size: 0.72rem !important; letter-spacing: 0.08em; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
[data-testid="stExpander"] details { border: 1px solid var(--border) !important; border-radius: 12px;
  background: var(--panel); }
[data-testid="stPlotlyChart"] { background: var(--panel); border: 1px solid var(--border);
  border-radius: 14px; padding: 6px 8px; }
[data-testid="stAlert"] { border-radius: 12px; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #f0b450, #c9822a);
  color: #111 !important; border: 0; font-weight: 700; border-radius: 10px; }
.cs-ticker { overflow: hidden; border: 1px solid var(--border); border-radius: 10px;
  background: var(--panel); margin: 0 0 1.1rem 0; white-space: nowrap; }
.cs-track { display: inline-flex; gap: 2.4rem; padding: 9px 0; animation: cs-scroll 40s linear infinite; }
.cs-ticker:hover .cs-track { animation-play-state: paused; }
.cs-item { font-family: var(--mono); font-size: 0.82rem; color: var(--text); }
.cs-item b { color: var(--muted); font-weight: 600; margin-right: .45rem; font-family: var(--sans);
  font-size: .72rem; letter-spacing: .08em; }
.cs-up { color: var(--up); } .cs-down { color: var(--down); } .cs-acc { color: var(--accent); }
@keyframes cs-scroll { from { transform: translateX(0); } to { transform: translateX(-50%%); } }
@media (prefers-reduced-motion: reduce) { .cs-track { animation: none; } }
.cs-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px;
  margin: .2rem 0 1rem; }
.cs-card { background: linear-gradient(180deg, var(--panel2), var(--panel)); border: 1px solid var(--border);
  border-radius: 14px; padding: 14px 16px; min-width: 0; }
.cs-card .l { color: var(--muted); font-size: .7rem; letter-spacing: .1em; text-transform: uppercase;
  font-weight: 600; }
.cs-card .v { font-family: var(--mono); font-size: 1.35rem; font-weight: 600; color: var(--text);
  margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.cs-card .s { color: var(--muted); font-size: .8rem; margin-top: 4px; }
.cs-badge { display: inline-block; font-family: var(--mono); font-size: .76rem; font-weight: 600;
  padding: 2px 8px; border-radius: 6px; margin-top: 6px; }
.cs-badge.up { background: rgba(22,199,132,.12); color: var(--up); }
.cs-badge.down { background: rgba(234,57,67,.12); color: var(--down); }
.cs-badge.flat { background: rgba(138,147,163,.14); color: var(--muted); }
.cs-hero { display: flex; flex-wrap: wrap; align-items: baseline; gap: .4rem 1rem; margin: .2rem 0 .8rem; }
.cs-hero .p { font-family: var(--mono); font-size: 2.6rem; font-weight: 700; letter-spacing: -0.02em;
  color: var(--text); }
.cs-hero .u { color: var(--muted); font-size: 1rem; }
.cs-pill { display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; padding: 2px 9px; border-radius: 999px; margin-left: 6px; }
</style>
"""
    % THEME
)


def apply_style() -> None:
    """Inject the terminal CSS (call once per page run, after ``set_page_config``)."""
    st.markdown(CSS, unsafe_allow_html=True)


def badge(delta_pct: float | None, suffix: str = "") -> str:
    """Coloured up/down percentage badge as HTML."""
    if delta_pct is None or delta_pct != delta_pct:  # None or NaN
        return ""
    if abs(delta_pct) < 0.05:
        cls, arrow = "flat", "■"
    else:
        cls, arrow = ("up", "▲") if delta_pct > 0 else ("down", "▼")
    return f'<span class="cs-badge {cls}">{arrow} {abs(delta_pct):.1f}%{html.escape(suffix)}</span>'


@dataclass
class Card:
    label: str
    value: str
    delta_pct: float | None = None
    sub: str = ""
    delta_suffix: str = ""
    pill: tuple[str, str] | None = None  # (text, colour)


def cards_html(items: list[Card]) -> str:
    """HTML for a responsive row of KPI cards."""
    parts = []
    for c in items:
        pill = ""
        if c.pill:
            text, colour = c.pill
            pill = (
                f'<span class="cs-pill" style="background:{colour}22;color:{colour}">'
                f"{html.escape(text)}</span>"
            )
        parts.append(
            '<div class="cs-card">'
            f'<div class="l">{html.escape(c.label)}</div>'
            f'<div class="v">{html.escape(c.value)}</div>'
            f"<div>{badge(c.delta_pct, c.delta_suffix)}{pill}</div>"
            f'<div class="s">{html.escape(c.sub)}</div>'
            "</div>"
        )
    return f'<div class="cs-cards">{"".join(parts)}</div>'


def cards(items: list[Card]) -> None:
    """Render a row of KPI cards."""
    st.markdown(cards_html(items), unsafe_allow_html=True)


def ticker_html(items: list[tuple[str, str, str]]) -> str:
    """HTML for the scrolling strip of (label, value, css class) items, repeated for a loop."""
    one = "".join(
        f'<span class="cs-item"><b>{html.escape(label)}</b>'
        f'<span class="{cls}">{html.escape(value)}</span></span>'
        for label, value, cls in items
    )
    return (
        f'<div class="cs-ticker" aria-label="Market summary"><div class="cs-track">{one}{one}'
        "</div></div>"
    )


def ticker(items: list[tuple[str, str, str]]) -> None:
    """Render the scrolling price strip (nothing if there are no items)."""
    if items:
        st.markdown(ticker_html(items), unsafe_allow_html=True)


def hero(price: str, unit: str, delta_pct: float | None, note: str) -> None:
    """Large headline price with its month-on-month change badge."""
    st.markdown(
        f'<div class="cs-hero"><span class="p">{html.escape(price)}</span>'
        f'<span class="u">{html.escape(unit)}</span>{badge(delta_pct, " m/m")}'
        f'<span class="u">{html.escape(note)}</span></div>',
        unsafe_allow_html=True,
    )
