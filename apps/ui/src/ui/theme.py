"""Brand theme + drop-in CSS for Deep Research.

Call ``inject_global_theme()`` exactly once near the top of ``main()`` in
``app.py`` (after ``st.set_page_config``). It loads the Inter font from
Google Fonts and installs the global chrome (semantic delta colors, tabular
numerals, card shadows, sidebar polish) used by the redesigned screens.

Also exposes a small ``COLORS`` palette + ``chart_color_sequence`` and
``chart_diverging_scale`` so Plotly/Altair charts inherit the same brand
colors that Streamlit widgets use via ``.streamlit/config.toml``.
"""

from __future__ import annotations

from typing import List, Tuple

import streamlit as st

COLORS = {
    "primary": "#1F3A68",
    "primary_soft": "#3E8FC1",
    "ink": "#1A2436",
    "muted": "#5B6776",
    "border": "#D6DBE4",
    "surface": "#FFFFFF",
    "surface_alt": "#F1F4F9",
    "good": "#1A8754",
    "bad": "#B5364B",
    "warn": "#C97B30",
    "info": "#0E7C7B",
    "violet": "#7A5BA1",
    "priority_high": "#B91C1C",
    "priority_med": "#B45309",
    "priority_low": "#047857",
}


def chart_color_sequence() -> List[str]:
    return [
        COLORS["primary"],
        COLORS["info"],
        COLORS["warn"],
        COLORS["violet"],
        COLORS["bad"],
        COLORS["primary_soft"],
        COLORS["muted"],
    ]


def chart_diverging_scale() -> List[Tuple[float, str]]:
    return [
        (0.0, "#B5364B"),
        (0.25, "#E2A1AB"),
        (0.5, "#F1F4F9"),
        (0.75, "#9DC2D6"),
        (1.0, "#1F3A68"),
    ]


def chart_sequential_scale() -> List[str]:
    return ["#EAF1F8", "#C7D8EC", "#A4BFE0", "#6B97C8", "#3D71AC", "#1F3A68"]


_GLOBAL_CSS = """
<style>
  /* Inter — graceful fallback to system sans if @import fails offline. */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stSidebar"] {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-feature-settings: "tnum" 1, "zero" 1, "ss02" 1;
  }
  code, pre, kbd { font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace; }

  /* Tighter, weightier headlines so KPI cards feel executive. */
  h1, h2, h3 { letter-spacing: -0.01em; font-weight: 700; color: #1A2436; }
  h4, h5, h6 { letter-spacing: -0.005em; font-weight: 600; color: #1A2436; }

  /* st.metric: big number, semantic delta. Cost-up = red (bad). */
  [data-testid="stMetricValue"] {
      font-size: 2.0rem;
      font-weight: 700;
      line-height: 1.1;
      color: #1A2436;
  }
  [data-testid="stMetricLabel"] {
      color: #5B6776;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 0.72rem;
  }
  [data-testid="stMetricDelta"] { font-weight: 600; }

  /* Bordered containers feel like cards. */
  div[data-testid="stVerticalBlockBorderWrapper"] {
      border-radius: 10px !important;
      box-shadow: 0 1px 2px rgba(15, 27, 45, 0.05);
      transition: box-shadow 0.15s ease;
  }
  div[data-testid="stVerticalBlockBorderWrapper"]:hover {
      box-shadow: 0 2px 6px rgba(15, 27, 45, 0.08);
  }

  /* Tabs: cleaner, brand-tinted underline with generous spacing so labels
     don't crowd each other (default gap is 4px which reads as one long run-on
     line for 6+ tabs). */
  div[data-baseweb="tab-list"] {
      gap: 24px;
      border-bottom: 1px solid #D6DBE4;
  }
  button[data-baseweb="tab"] {
      font-weight: 600;
      letter-spacing: 0.01em;
      color: #5B6776;
      padding: 10px 14px;
  }
  button[data-baseweb="tab"]:hover {
      color: #1A2436;
      background: rgba(31, 58, 104, 0.04);
      border-radius: 6px 6px 0 0;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
      color: #1F3A68;
  }

  /* Pills and badges read more like chips. */
  [data-testid="stPills"] button[kind="pillsInactive"],
  [data-testid="stPills"] button[kind="pillsActive"] {
      border-radius: 999px;
      font-weight: 500;
  }

  /* Generous breathing room around the page body. */
  section.main > div.block-container {
      padding-top: 2rem;
      padding-bottom: 4rem;
      max-width: 1400px;
  }

  /* Sidebar: brand-dark feel. */
  [data-testid="stSidebar"] { color: #E2E8F0; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] label { color: #E2E8F0 !important; }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #E2E8F0; }
</style>
"""


def inject_global_theme() -> None:
    """Inject the global CSS once per page render."""

    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
