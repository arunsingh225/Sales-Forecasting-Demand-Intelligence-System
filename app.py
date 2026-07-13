import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 1.6rem; font-weight: 700; color: #ffffff;
    }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #aaaaaa; }
    [data-testid="metric-container"] {
        background-color: #1e1e2e; border: 1px solid #3d3d5c;
        border-radius: 10px; padding: 15px;
    }
    h1 { color: #ffffff; }
    h2 { color: #e0e0e0; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
    h3 { color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HELPER — bulletproof JSON → DataFrame (replaces pd.read_json)
# ════════════════════════════════════════════════════════════
def json_to_df(json_str):
    """
    100% pandas-version-safe JSON parser.
    pd.read_json(string) crashes on pandas 3.x — this never does.
    """
    return pd.DataFrame(json.loads(json_str))


# ════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_and_prep():
    df = pd.read_csv('train.csv', encoding='latin1')
    df.columns = df.columns.str.strip()
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
    df['Year']       = df['Order Date'].dt.year
    df['Month']      = df['Order Date'].dt.month
    df['Quarter']    = df['Order Date'].dt.quarter
    df['Ship_Days']  = (df['Ship Date'] - df['Order Date']).dt.days

    monthly = (df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales']
                 .sum().reset_index()
                 .rename(columns={'Order Date': 'Date', 'Sales': 'Monthly_Sales'}))

    weekly = (df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales']
                .sum().reset_index()
                .rename(columns={'Order Date': 'Date', 'Sales': 'Sales'}))

    return df, monthly, weekly


# ════════════════════════════════════════════════════════════
# CACHED FUNCTIONS
# ════════════════════════════════════════════════════════════
@st.cache_data
def run_prophet(series_json, horizon):
    # ── FIX: json.loads instead of pd.read_json (pandas 3.x safe) ──
    df_p = json_to_df(series_json)
    df_p['ds'] = pd.to_datetime(df_p['ds'])
    df_p = df_p.sort_values('ds').reset_index(drop=True)

    train_df = df_p.iloc[:-3]
    test_df  = df_p.iloc[-3:]

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode='additive',
                changepoint_prior_scale=0.1)
    m.fit(train_df)

    # ── FIX: freq compatibility — try 'ME' first, fall back to 'M' ──
    def make_future(model, periods):
        for freq in ['ME', 'M']:
            try:
                return model.make_future_dataframe(periods=periods, freq=freq)
            except Exception:
                continue
        # Last resort: generate dates manually
        last = model.history_dates.max()
        dates_hist = pd.DataFrame({'ds': model.history_dates})
        dates_fut  = pd.DataFrame({'ds': pd.date_range(last, periods=periods+1, freq='MS')[1:]})
        return pd.concat([dates_hist, dates_fut], ignore_index=True)

    fc_test   = m.predict(make_future(m, 3))
    test_pred = fc_test.tail(3)['yhat'].values

    mae  = mean_absolute_error(test_df['y'].values, test_pred)
    rmse = np.sqrt(mean_squared_error(test_df['y'].values, test_pred))

    fc = m.predict(make_future(m, horizon))
    return fc, mae, rmse, df_p


@st.cache_data
def run_anomaly(weekly_json):
    # ── FIX: json.loads instead of pd.read_json ──
    ws = json_to_df(weekly_json)
    ws.columns = ['Date', 'Sales']
    ws['Date'] = pd.to_datetime(ws['Date'])

    iso = IsolationForest(contamination=0.05, random_state=42)
    ws['iso_anomaly'] = iso.fit_predict(ws[['Sales']]) == -1

    ws['rolling_mean'] = ws['Sales'].rolling(4, min_periods=1).mean()
    ws['rolling_std']  = ws['Sales'].rolling(4, min_periods=1).std()
    ws['z_score']      = (ws['Sales'] - ws['rolling_mean']) / ws['rolling_std']
    ws['z_anomaly']    = ws['z_score'].abs() > 2
    ws['either']       = ws['iso_anomaly'] | ws['z_anomaly']
    return ws


@st.cache_data
def run_clustering(df_json):
    # ── FIX: json.loads instead of pd.read_json ──
    df = json_to_df(df_json)

    agg = df.groupby('Sub-Category').agg(
        Total_Sales   = ('Sales', 'sum'),
        Avg_Order_Val = ('Sales', 'mean'),
        Volatility    = ('Sales', 'std'),
        Order_Count   = ('Sales', 'count')
    ).reset_index().fillna(0)

    yearly = df.groupby(['Sub-Category', 'Year'])['Sales'].sum().reset_index()

    def calc_growth(g):
        s = g.sort_values('Year')
        return ((s.iloc[-1]['Sales'] - s.iloc[0]['Sales']) / s.iloc[0]['Sales'] * 100
                if len(s) >= 2 else 0)

    gr = yearly.groupby('Sub-Category').apply(
        calc_growth, include_groups=False).reset_index()
    gr.columns = ['Sub-Category', 'Growth_Rate']

    feat   = agg.merge(gr, on='Sub-Category')
    fcols  = ['Total_Sales', 'Avg_Order_Val', 'Volatility', 'Growth_Rate']
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(feat[fcols])

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    feat['Cluster'] = km.fit_predict(X_s)

    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_), columns=fcols)

    def assign_label(row):
        hs = row['Total_Sales']  >= centroids['Total_Sales'].median()
        hg = row['Growth_Rate']  >= centroids['Growth_Rate'].median()
        hv = row['Volatility']   >= centroids['Volatility'].median()
        if hs and not hv:   return 'High Volume, Stable Demand'
        elif hg and not hs: return 'Growing Demand'
        elif hv and not hs: return 'Low Volume, High Volatility'
        else:               return 'Declining Demand'

    label_map = {i: lbl for i, lbl in
                 enumerate(centroids.apply(assign_label, axis=1))}
    feat['Cluster_Label'] = feat['Cluster'].map(label_map)

    pca = PCA(n_components=2)
    Xp  = pca.fit_transform(X_s)
    feat['PCA1'] = Xp[:, 0]
    feat['PCA2'] = Xp[:, 1]
    return feat, pca


# ── Load Data ─────────────────────────────────────────────────
df, monthly_sales, weekly_sales = load_and_prep()


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
st.sidebar.markdown("## 📊 Sales Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate to", [
    "🏠 Sales Overview",
    "🔮 Forecast Explorer",
    "⚠️ Anomaly Report",
    "🗂️ Product Segments"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Superstore Sales")
st.sidebar.markdown(f"**Records:** {df.shape[0]:,}")
st.sidebar.markdown(
    f"**Period:** {df['Order Date'].min().strftime('%b %Y')} → "
    f"{df['Order Date'].max().strftime('%b %Y')}")
st.sidebar.markdown("---")
st.sidebar.caption("Built by Arun | XYlofy AI Internship")


# ════════════════════════════════════════════════════════════
# PAGE 1 — SALES OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏠 Sales Overview":
    st.title("🏠 Sales Overview Dashboard")
    st.markdown("*4-year retail sales analysis across categories and regions*")
    st.markdown("---")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        sel_regions = st.multiselect("Filter by Region",
                                     options=df['Region'].unique().tolist(),
                                     default=df['Region'].unique().tolist())
    with col_f2:
        sel_cats = st.multiselect("Filter by Category",
                                  options=df['Category'].unique().tolist(),
                                  default=df['Category'].unique().tolist())

    fdf = df[df['Region'].isin(sel_regions) & df['Category'].isin(sel_cats)]

    if fdf.empty:
        st.warning("⚠️ No data for selected filters.")
        st.stop()

    st.markdown("### 📌 Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Total Revenue",   f"${fdf['Sales'].sum():,.0f}")
    k2.metric("📦 Total Orders",    f"{fdf.shape[0]:,}")
    k3.metric("🛒 Avg Order Value", f"${fdf['Sales'].mean():.2f}")
    k4.metric("🚚 Avg Ship Days",   f"{fdf['Ship_Days'].mean():.1f} days")
    st.markdown("---")

    st.markdown("### 📅 Total Sales by Year")
    yr = fdf.groupby('Year')['Sales'].sum()
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax1.bar(yr.index, yr.values, color=colors[:len(yr)],
                   edgecolor='black', width=0.5)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Total Sales ($)', fontsize=12)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1000,
                 f'${bar.get_height():,.0f}',
                 ha='center', fontsize=10, fontweight='bold')
    if yr.max() > 0:
        ax1.set_ylim(0, yr.max() * 1.15)
    plt.tight_layout()
    st.pyplot(fig1); plt.close()

    st.markdown("### 📈 Monthly Sales Trend")
    monthly_f = (fdf.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales']
                    .sum().reset_index())
    fig2, ax2 = plt.subplots(figsize=(13, 4))
    ax2.plot(monthly_f['Order Date'], monthly_f['Sales'],
             color='#2c3e50', lw=1.8, marker='o', markersize=3)
    ax2.fill_between(monthly_f['Order Date'], monthly_f['Sales'],
                     alpha=0.1, color='#3498db')
    ax2.set_xlabel('Date'); ax2.set_ylabel('Monthly Sales ($)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig2); plt.close()

    st.markdown("### 📊 Sales Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**By Category**")
        cat_s = fdf.groupby('Category')['Sales'].sum().sort_values()
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        b3 = ax3.barh(cat_s.index, cat_s.values,
                      color=['#2ecc71','#3498db','#e74c3c'][:len(cat_s)],
                      edgecolor='black')
        for bar in b3:
            ax3.text(bar.get_width()*0.97, bar.get_y()+bar.get_height()/2,
                     f'${bar.get_width():,.0f}', va='center', ha='right',
                     fontsize=9, color='white', fontweight='bold')
        ax3.set_xlabel('Sales ($)')
        ax3.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        if len(cat_s) > 0: ax3.set_xlim(0, cat_s.max()*1.05)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    with c2:
        st.markdown("**By Region**")
        reg_s = fdf.groupby('Region')['Sales'].sum().sort_values()
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        b4 = ax4.barh(reg_s.index, reg_s.values,
                      color=['#9b59b6','#f39c12','#1abc9c','#e67e22'][:len(reg_s)],
                      edgecolor='black')
        for bar in b4:
            ax4.text(bar.get_width()*0.97, bar.get_y()+bar.get_height()/2,
                     f'${bar.get_width():,.0f}', va='center', ha='right',
                     fontsize=9, color='white', fontweight='bold')
        ax4.set_xlabel('Sales ($)')
        ax4.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        if len(reg_s) > 0: ax4.set_xlim(0, reg_s.max()*1.05)
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    st.markdown("### 🏆 Top 10 Sub-Categories by Revenue")
    top10 = (fdf.groupby('Sub-Category')['Sales'].sum()
               .sort_values(ascending=False).head(10).reset_index())
    top10.columns = ['Sub-Category', 'Total Sales ($)']
    top10['Total Sales ($)'] = top10['Total Sales ($)'].round(2)
    top10.insert(0, 'Rank', range(1, len(top10)+1))
    st.dataframe(top10, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST EXPLORER
# ════════════════════════════════════════════════════════════
elif page == "🔮 Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.markdown("*Prophet-based sales forecasting for categories and regions*")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        all_cats = sorted(df['Category'].unique().tolist())
        sel_cats = st.multiselect("Filter by Category", options=all_cats, default=all_cats)
    with col2:
        all_regs = sorted(df['Region'].unique().tolist())
        sel_regs = st.multiselect("Filter by Region", options=all_regs, default=all_regs)
    with col3:
        horizon = st.slider("Months Ahead", min_value=1, max_value=3, value=3)

    if not sel_cats or not sel_regs:
        st.warning("⚠️ Select at least one Category and one Region to continue.")
        st.stop()

    filt = df[df['Category'].isin(sel_cats) & df['Region'].isin(sel_regs)]

    if len(sel_cats) == len(all_cats) and len(sel_regs) == len(all_regs):
        seg_name = "Overall"
    elif len(sel_cats) == 1 and len(sel_regs) == 1:
        seg_name = f"{sel_cats[0]} — {sel_regs[0]}"
    elif len(sel_cats) == 1:
        seg_name = f"{sel_cats[0]} ({', '.join(sel_regs)})"
    elif len(sel_regs) == 1:
        seg_name = f"{sel_regs[0]} ({', '.join(sel_cats)})"
    else:
        seg_name = f"{', '.join(sel_cats)} | {', '.join(sel_regs)}"

    series = (filt.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales']
                  .sum().reset_index()
                  .rename(columns={'Order Date': 'ds', 'Sales': 'y'}))

    run_btn = st.button("🚀 Run Forecast", use_container_width=True)

    if run_btn:
        try:
            with st.spinner(f"Fitting Prophet model for {seg_name}..."):
                series_json = series.to_json(date_format='iso', orient='records')
                fc, mae, rmse, hist = run_prophet(series_json, horizon)
        except Exception as e:
            import traceback
            st.error(f"❌ Forecast failed: {e}")
            st.code(traceback.format_exc(), language='text')
            st.stop()

        st.markdown("---")
        st.markdown(f"### 📈 Forecast: **{seg_name}** — Next {horizon} Month(s)")

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(hist['ds'], hist['y'], color='#2c3e50', lw=1.8,
                label='Historical Sales')
        future_rows = fc.tail(horizon)
        ax.plot(future_rows['ds'], future_rows['yhat'],
                color='#e74c3c', lw=2.5, marker='o', ls='--', label='Forecast')
        ax.fill_between(future_rows['ds'],
                        future_rows['yhat_lower'], future_rows['yhat_upper'],
                        alpha=0.2, color='#e74c3c', label='95% Confidence Interval')
        ax.axvline(hist['ds'].max(), color='gray', lw=1, ls=':', alpha=0.7)
        ax.set_xlim(hist['ds'].min(),
                    future_rows['ds'].max() + pd.DateOffset(months=1))
        ax.set_xlabel('Month'); ax.set_ylabel('Sales ($)')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=30); plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("### 📋 Forecast Values")
        fc_table = future_rows[['ds','yhat','yhat_lower','yhat_upper']].copy()
        fc_table.columns = ['Month','Forecast ($)','Lower Bound ($)','Upper Bound ($)']
        fc_table['Month'] = fc_table['Month'].dt.strftime('%b %Y')
        for c in ['Forecast ($)','Lower Bound ($)','Upper Bound ($)']:
            fc_table[c] = fc_table[c].round(2)
        st.dataframe(fc_table, use_container_width=True, hide_index=True)

        st.markdown("### 🎯 Model Accuracy (last 3 months holdout)")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"${mae:,.2f}")
        m2.metric("RMSE", f"${rmse:,.2f}")
        m3.metric("Model", "Prophet")
        st.info("💡 Lower MAE & RMSE = better accuracy.")
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px; color:#888;'>
            <h3>👆 Select segment and click Run Forecast</h3>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY REPORT
# ════════════════════════════════════════════════════════════
elif page == "⚠️ Anomaly Report":
    st.title("⚠️ Anomaly Report")
    st.markdown("*Detecting unusual sales weeks using Isolation Forest & Z-Score*")
    st.markdown("---")

    with st.spinner("Running anomaly detection..."):
        ws = run_anomaly(
            weekly_sales.to_json(date_format='iso', orient='records'))

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Weeks",            f"{len(ws)}")
    k2.metric("Isolation Forest Flags", f"{ws['iso_anomaly'].sum()}")
    k3.metric("Z-Score Flags",          f"{ws['z_anomaly'].sum()}")
    st.markdown("---")

    st.markdown("### 🔍 Method 1 — Isolation Forest")
    fig1, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(ws['Date'], ws['Sales'], color='#2c3e50', lw=1.2, label='Weekly Sales')
    a_iso = ws[ws['iso_anomaly']]
    ax1.scatter(a_iso['Date'], a_iso['Sales'], color='#e74c3c',
                s=80, zorder=5, marker='v', label='Anomaly')
    ax1.set_xlabel('Date'); ax1.set_ylabel('Sales ($)'); ax1.legend()
    plt.tight_layout(); st.pyplot(fig1); plt.close()

    st.markdown("### 📐 Method 2 — Z-Score (±2σ)")
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(ws['Date'], ws['Sales'], color='#2c3e50', lw=1.2, label='Weekly Sales')
    ax2.plot(ws['Date'], ws['rolling_mean'], color='#3498db',
             lw=1.5, ls='--', label='Rolling Mean')
    a_z = ws[ws['z_anomaly']]
    ax2.scatter(a_z['Date'], a_z['Sales'], color='#9b59b6',
                s=80, zorder=5, marker='^', label='Anomaly')
    ax2.set_xlabel('Date'); ax2.set_ylabel('Sales ($)'); ax2.legend()
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    st.markdown("### 📋 Detected Anomaly Weeks")
    mean_s = ws['Sales'].mean()

    def get_explanation(row):
        month = row['Date'].month
        diff  = ((row['Sales'] - mean_s) / mean_s) * 100
        if month in [11, 12]: return "Festive season / Year-end bulk orders"
        elif month in [1, 2]: return "Post-holiday demand slowdown"
        elif month in [7, 8]: return "Mid-year sale / Back-to-school demand"
        elif diff > 50:        return "Large corporate/bulk order received"
        elif diff < -40:       return "Supply disruption or stockout"
        else:                  return "Promotional campaign or regional event"

    all_anom = ws[ws['either']].copy()
    all_anom['Type']        = all_anom['Sales'].apply(
        lambda x: "📈 Spike" if x > mean_s else "📉 Drop")
    all_anom['Explanation'] = all_anom.apply(get_explanation, axis=1)
    all_anom['IsoForest']   = all_anom['iso_anomaly'].map({True:'✅', False:'❌'})
    all_anom['Z-Score']     = all_anom['z_anomaly'].map({True:'✅', False:'❌'})
    all_anom['Week']        = all_anom['Date'].dt.strftime('%d %b %Y')
    all_anom['Sales ($)']   = all_anom['Sales'].round(2)
    display_anom = all_anom[['Week','Sales ($)','Type',
                              'IsoForest','Z-Score','Explanation']].copy()
    display_anom.columns = ['Week','Sales ($)','Type',
                             'Isolation Forest','Z-Score','Likely Cause']
    st.dataframe(display_anom.reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    both = ws[ws['iso_anomaly'] & ws['z_anomaly']]
    st.info(f"📌 **{len(both)} weeks flagged by BOTH methods** — "
            f"these are the most reliable anomalies.")


# ════════════════════════════════════════════════════════════
# PAGE 4 — PRODUCT SEGMENTS
# ════════════════════════════════════════════════════════════
elif page == "🗂️ Product Segments":
    st.title("🗂️ Product Demand Segments")
    st.markdown("*K-Means clustering of sub-categories by demand behavior*")
    st.markdown("---")

    with st.spinner("Running clustering analysis..."):
        df_slim = df[['Sub-Category', 'Sales', 'Year']].copy()
        feat_df, pca = run_clustering(df_slim.to_json(orient='records'))

    cluster_counts = feat_df['Cluster_Label'].value_counts()
    colors_kpi = {
        'High Volume, Stable Demand'  : '🟢',
        'Growing Demand'              : '🔵',
        'Low Volume, High Volatility' : '🔴',
        'Declining Demand'            : '🟡'
    }
    cols = st.columns(max(len(cluster_counts), 1))
    for i, (label, count) in enumerate(cluster_counts.items()):
        cols[i].metric(f"{colors_kpi.get(label,'⚪')} {label}",
                       f"{count} sub-categories")

    st.markdown("---")
    st.markdown("### 🔵 Demand Cluster Map (PCA View)")

    import plotly.graph_objects as go

    colors_map = {
        'High Volume, Stable Demand'  : '#2ecc71',
        'Low Volume, High Volatility' : '#e74c3c',
        'Growing Demand'              : '#3498db',
        'Declining Demand'            : '#f39c12'
    }

    _pos_cycle = {}
    def get_textpos(pca1, pca2, key):
        if pca1 >= 0 and pca2 >= 0:
            opts = ['top right', 'middle right', 'top center']
        elif pca1 >= 0 and pca2 < 0:
            opts = ['bottom right', 'middle right', 'bottom center']
        elif pca1 < 0 and pca2 >= 0:
            opts = ['top left', 'middle left', 'top center']
        else:
            opts = ['bottom left', 'middle left', 'bottom center']
        _pos_cycle[key] = _pos_cycle.get(key, -1) + 1
        return opts[_pos_cycle[key] % len(opts)]

    fig_pca = go.Figure()

    for label, group in feat_df.groupby('Cluster_Label'):
        fig_pca.add_trace(go.Scatter(
            x=group['PCA1'],
            y=group['PCA2'],
            mode='markers+text',
            name=label,
            text=group['Sub-Category'],
            textposition=[get_textpos(r['PCA1'], r['PCA2'], f"{label}-{idx}")
                          for idx, r in group.iterrows()],
            textfont=dict(size=9, color='#2c3e50'),
            marker=dict(
                color=colors_map.get(label, '#95a5a6'),
                size=12,
                line=dict(color='black', width=1),
                opacity=0.85
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "PC1: %{x:.2f}<br>"
                "PC2: %{y:.2f}<br>"
                f"Cluster: {label}<extra></extra>"
            )
        ))

    fig_pca.add_hline(y=0, line_dash='dot', line_color='gray',
                      line_width=0.8, opacity=0.5)
    fig_pca.add_vline(x=0, line_dash='dot', line_color='gray',
                      line_width=0.8, opacity=0.5)

    fig_pca.update_layout(
        title=dict(
            text='Product Demand Segmentation — PCA View',
            font=dict(size=15, color='#1a1a1a'), x=0.5
        ),
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)",
        height=580,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            yanchor='top', y=0.99,
            xanchor='right', x=0.99,
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#cccccc', borderwidth=1,
            font=dict(color='#1a1a1a', size=11)
        ),
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Sub-Category Assignments")
    stocking = {
        'High Volume, Stable Demand'  : 'Bulk stock | 30-day safety buffer',
        'Low Volume, High Volatility' : 'Just-in-time | Avoid overstock',
        'Growing Demand'              : 'Pre-stock 2 months ahead',
        'Declining Demand'            : 'Reduce stock | Run promotions to clear'
    }
    sel_cluster = st.selectbox("Filter by Cluster",
        ['All'] + sorted(feat_df['Cluster_Label'].unique().tolist()))

    display = feat_df[['Sub-Category','Cluster_Label',
                        'Total_Sales','Growth_Rate']].copy()
    display['Total_Sales']       = display['Total_Sales'].round(2)
    display['Growth_Rate']       = display['Growth_Rate'].round(2)
    display['Stocking Strategy'] = display['Cluster_Label'].map(stocking)
    display.columns = ['Sub-Category','Demand Cluster','Total Sales ($)',
                       'Growth Rate (%)','Stocking Strategy']
    display = display.sort_values('Demand Cluster')
    if sel_cluster != 'All':
        display = display[display['Demand Cluster'] == sel_cluster]
    st.dataframe(display.reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🏪 Recommended Stocking Strategy")
    s1, s2 = st.columns(2)
    with s1:
        st.success("🟢 **High Volume, Stable Demand**\n\nBulk safety stock. "
                   "30-day reorder point. Long-term supplier contracts.")
        st.info("🔵 **Growing Demand**\n\nPre-stock aggressively 2 months ahead. "
                "Prioritize shelf space and supplier allocation.")
    with s2:
        st.error("🔴 **Low Volume, High Volatility**\n\nMinimal safety stock. "
                 "Just-in-time procurement only.")
        st.warning("🟡 **Declining Demand**\n\nReduce stock gradually. "
                   "Run clearance promotions. Avoid replenishment.")
