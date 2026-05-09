"""Pastel theme — milk-pink palette dùng chung cho toàn dự án."""
import streamlit as st

PALETTE = {
    "milk_pink":     "#FFF5F7",
    "milk_pink_2":   "#FFEAF1",
    "rose":          "#F7B8C9",
    "rose_dark":     "#E48BA7",
    "lavender":      "#D7C4F2",
    "lavender_dark": "#A993E0",
    "mint":          "#BFE7D6",
    "mint_dark":     "#7FCBA9",
    "peach":         "#FFD6B8",
    "peach_dark":    "#F5A678",
    "sky":           "#C7E3F5",
    "sky_dark":      "#7FB6DC",
    "cream":         "#FFF7E6",
    "ink":           "#5B4A55",
    "ink_soft":      "#8C7785",
    "danger":        "#E48BA7",
    "warning":       "#F5C77E",
    "success":       "#7FCBA9",
}

PALETTE_LIST = [
    PALETTE["rose"], PALETTE["lavender"], PALETTE["mint"],
    PALETTE["peach"], PALETTE["sky"], PALETTE["cream"],
    PALETTE["rose_dark"], PALETTE["lavender_dark"],
    PALETTE["mint_dark"], PALETTE["peach_dark"], PALETTE["sky_dark"],
]


def inject_pastel_css():
    """Bơm CSS pastel vào trang hiện tại. Gọi 1 lần trong mỗi page."""
    st.markdown(
        f"""
<style>
/* ── Nền tổng thể ─────────────────────────────── */
.stApp {{
    background: linear-gradient(180deg,{PALETTE['milk_pink']} 0%,{PALETTE['milk_pink_2']} 100%);
    color: {PALETTE['ink']};
}}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,#FFE4EC 0%,#F5E6FF 100%);
}}
section[data-testid="stSidebar"] * {{ color:{PALETTE['ink']} !important; }}

/* ── Heading ──────────────────────────────────── */
h1,h2,h3,h4,h5 {{ color:{PALETTE['ink']}; font-family:"Segoe UI",system-ui,sans-serif; }}
h1 {{ letter-spacing:.5px; }}

/* ── Buttons ──────────────────────────────────── */
.stButton>button, .stDownloadButton>button {{
    background: linear-gradient(135deg,{PALETTE['rose']} 0%,{PALETTE['lavender']} 100%);
    color:{PALETTE['ink']};
    border:none; border-radius:14px;
    padding:.55em 1.4em; font-weight:600;
    box-shadow:0 4px 14px rgba(228,139,167,.25);
    transition:all .2s ease;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
    transform:translateY(-2px);
    box-shadow:0 8px 22px rgba(228,139,167,.35);
}}
.stButton>button:focus {{ outline:none; box-shadow:0 0 0 3px rgba(228,139,167,.3); }}

/* ── Tabs ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background:#FFFAFC; border-radius:18px; padding:6px;
    box-shadow:inset 0 0 0 1px rgba(228,139,167,.18);
    gap:4px;
}}
.stTabs [data-baseweb="tab"] {{
    background:transparent; color:{PALETTE['ink_soft']};
    border-radius:14px; padding:10px 18px; font-weight:600;
    transition:all .15s ease;
}}
.stTabs [aria-selected="true"] {{
    background:linear-gradient(135deg,{PALETTE['rose']} 0%,{PALETTE['peach']} 100%) !important;
    color:{PALETTE['ink']} !important;
    box-shadow:0 4px 12px rgba(228,139,167,.25);
}}

/* ── Inputs / Select / Slider ─────────────────── */
.stTextInput>div>div>input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"]>div,
.stDateInput input, .stTimeInput input,
textarea {{
    background:#FFFAFC !important;
    border:1px solid #F4D7E1 !important;
    border-radius:12px !important;
    color:{PALETTE['ink']} !important;
}}
.stSlider [data-baseweb="slider"] > div > div {{
    background:{PALETTE['rose']} !important;
}}

/* ── Metric / Info / Warning / Success ─────────── */
[data-testid="stMetric"] {{
    background:#FFFAFC; border:1px solid #F4D7E1;
    border-radius:16px; padding:14px 18px;
    box-shadow:0 2px 8px rgba(228,139,167,.08);
}}
[data-testid="stMetricValue"] {{ color:{PALETTE['rose_dark']}; }}

div[data-testid="stAlert"] {{ border-radius:14px; border:none; }}
div[data-testid="stAlert"][kind="info"]    {{ background:#EAF4FB; }}
div[data-testid="stAlert"][kind="success"] {{ background:#E7F7EF; }}
div[data-testid="stAlert"][kind="warning"] {{ background:#FFF4DE; }}
div[data-testid="stAlert"][kind="error"]   {{ background:#FCE4EC; }}

/* ── Dataframe ────────────────────────────────── */
[data-testid="stDataFrame"] {{
    background:#FFFAFC; border-radius:14px; padding:4px;
    box-shadow:0 2px 8px rgba(228,139,167,.06);
}}

/* ── Expander ─────────────────────────────────── */
.streamlit-expanderHeader, [data-testid="stExpander"] details summary {{
    background:#FFEFF4 !important; border-radius:12px !important;
    color:{PALETTE['ink']} !important;
}}

/* ── Hero ─────────────────────────────────────── */
.hero-title {{
    font-size:54px; font-weight:800; text-align:center;
    background:linear-gradient(90deg,{PALETTE['rose_dark']},{PALETTE['lavender_dark']},{PALETTE['peach_dark']});
    -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color:transparent;
    letter-spacing:1px; margin:.1em 0 .15em;
    text-shadow:0 4px 20px rgba(228,139,167,.15);
}}
.hero-subtitle {{
    text-align:center; font-size:20px; color:{PALETTE['ink_soft']};
    margin-bottom:1.5em; font-weight:500;
}}
.hero-tag {{
    display:inline-block; background:#FFEAF1; color:{PALETTE['rose_dark']};
    padding:6px 14px; border-radius:999px; font-size:13px; font-weight:600;
    border:1px solid #F4D7E1;
}}

/* ── Card ─────────────────────────────────────── */
.pastel-card {{
    background:#FFFAFC; border-radius:18px; padding:20px;
    box-shadow:0 4px 18px rgba(228,139,167,.10);
    border:1px solid #F8E2EA; margin-bottom:14px;
}}
.pastel-card h3 {{ margin-top:0; color:{PALETTE['rose_dark']}; }}

/* ── Badge ─────────────────────────────────────── */
.badge {{
    display:inline-block; padding:14px 22px; border-radius:14px;
    font-size:22px; font-weight:700; text-align:center; width:100%;
    box-shadow:0 4px 14px rgba(0,0,0,.06);
}}
.badge-ok    {{ background:#E7F7EF; color:#3F8D6F; border:2px solid {PALETTE['mint_dark']}; }}
.badge-warn  {{ background:#FFF4DE; color:#A8771A; border:2px solid {PALETTE['peach_dark']}; }}
.badge-block {{ background:#FCE4EC; color:#A85070; border:2px solid {PALETTE['rose_dark']}; }}

/* ── Chat bubble ──────────────────────────────── */
.chat-user {{
    background:{PALETTE['lavender']}; color:{PALETTE['ink']};
    padding:10px 14px; border-radius:14px 14px 4px 14px;
    margin:6px 0; max-width:85%; margin-left:auto;
}}
.chat-bot {{
    background:#FFFAFC; color:{PALETTE['ink']};
    padding:10px 14px; border-radius:14px 14px 14px 4px;
    margin:6px 0; max-width:85%;
    border:1px solid #F4D7E1;
}}

/* ── Hide default Streamlit footer/header ─────── */
#MainMenu, footer {{ visibility:hidden; }}
</style>
        """,
        unsafe_allow_html=True,
    )


def matplotlib_pastel():
    """Áp pastel cho matplotlib (gọi mỗi lần vẽ)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.facecolor":   "#FFFAFC",
        "figure.facecolor": "#FFFAFC",
        "axes.edgecolor":   "#E4C8D2",
        "axes.labelcolor":  PALETTE["ink"],
        "xtick.color":      PALETTE["ink_soft"],
        "ytick.color":      PALETTE["ink_soft"],
        "axes.titlecolor":  PALETTE["ink"],
        "axes.grid":        True,
        "grid.color":       "#F4D7E1",
        "grid.alpha":       .5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def plotly_pastel_layout(fig, height=380):
    """Áp pastel cho plotly figure tại chỗ."""
    fig.update_layout(
        paper_bgcolor="#FFFAFC",
        plot_bgcolor="#FFFAFC",
        font=dict(color=PALETTE["ink"], family="Segoe UI, system-ui"),
        xaxis=dict(gridcolor="#F4D7E1", zerolinecolor="#F4D7E1"),
        yaxis=dict(gridcolor="#F4D7E1", zerolinecolor="#F4D7E1"),
        height=height,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig
