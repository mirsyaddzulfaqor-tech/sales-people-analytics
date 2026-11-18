"""Chart palette and styling.

Colours come from a categorical palette validated for colour-vision
deficiency (CVD). Slots 1-3 are safe for every chart form including scatter;
beyond three categories, use the status palette or fold the remainder into
"Other".
"""

# --- Categorical (fixed order, never cycled) ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
S1, S2, S3 = SERIES[0], SERIES[1], SERIES[2]

# --- Status (never reused as an ordinary series colour) ---
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Risk tier -> colour + icon. Colour never stands alone: a text label always
# accompanies it, per the accessibility rule.
RISK_STYLE = {
    "Low":      (STATUS["good"], "●"),
    "Moderate": (STATUS["warning"], "▲"),
    "High":     (STATUS["serious"], "▲"),
    "Critical": (STATUS["critical"], "■"),
}

# --- Single-hue sequential ramp (light to dark blue) ---
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# --- Diverging: blue <-> red, neutral grey midpoint ---
DIVERGING = [[0.0, "#184f95"], [0.5, "#f0efec"], [1.0, "#d03b3b"]]

# --- Chart chrome and ink (light surface) ---
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def apply_layout(fig, height=360, showlegend=True, ylab=None, xlab=None):
    """Apply consistent chart chrome to a Plotly figure."""
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK_SECONDARY),
        title_font=dict(family=FONT, size=15, color=INK_PRIMARY),
        margin=dict(l=8, r=8, t=48, b=8),
        hoverlabel=dict(font_family=FONT, font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    title_text="", font=dict(size=12)),
    )
    fig.update_xaxes(showgrid=False, linecolor=BASELINE, ticks="outside",
                     tickcolor=BASELINE, title_text=xlab or "",
                     title_font=dict(size=12, color=INK_MUTED),
                     tickfont=dict(size=11, color=INK_MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False,
                     linecolor="rgba(0,0,0,0)", title_text=ylab or "",
                     title_font=dict(size=12, color=INK_MUTED),
                     tickfont=dict(size=11, color=INK_MUTED))
    return fig


def idr(x, unit="auto"):
    """Format an IDR amount so a human can read it at a glance."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if unit == "million" or (unit == "auto" and x < 1_000_000_000):
        return f"{sign}Rp{x / 1_000_000:,.1f}M"
    return f"{sign}Rp{x / 1_000_000_000:,.2f}B"
