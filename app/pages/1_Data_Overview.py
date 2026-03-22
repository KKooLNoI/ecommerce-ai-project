import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #060A10 !important;
    font-family: 'DM Sans', sans-serif; color: #94a3b8;
}
[data-testid="stSidebar"] { background: #0B1120 !important; border-right: 1px solid rgba(255,255,255,0.05); }
section.main > div { padding: 2rem 2.5rem; }
[data-testid="stDecoration"], header { display: none !important; }

.page-header { padding: 1.5rem 0 2rem; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 2.5rem; }
.page-eyebrow { font-size:.68rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; color:#f97316; margin-bottom:.5rem; }
.page-title { font-family:'Playfair Display',serif; font-size:2.2rem; font-weight:800; color:#f1f5f9; margin-bottom:.5rem; }
.page-sub { font-size:.9rem; color:#475569; }

.metric-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:1rem; margin-bottom:2.5rem; }
.metric-box {
    background:#0d1627; border:1px solid rgba(255,255,255,0.05);
    border-radius:16px; padding:1.3rem; text-align:center;
}
.metric-val { font-family:'Playfair Display',serif; font-size:1.7rem; font-weight:700; color:#f97316; }
.metric-label { font-size:.72rem; color:#334155; margin-top:.2rem; letter-spacing:.04em; }

.panel {
    background:#0d1627; border:1px solid rgba(255,255,255,0.06);
    border-radius:18px; padding:1.8rem; margin-bottom:1.5rem;
}
.panel-title { font-size:.68rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:#475569; margin-bottom:1.2rem; display:flex; align-items:center; gap:.5rem; }
.panel-title::after { content:''; flex:1; height:1px; background:rgba(255,255,255,0.04); }

.insight-box {
    background:rgba(249,115,22,0.05); border:1px solid rgba(249,115,22,0.15);
    border-radius:12px; padding:1rem 1.2rem; margin-top:1rem;
    font-size:.85rem; color:#94a3b8; line-height:1.7;
}
.insight-box b { color:#f97316; }

label { color:#64748b !important; font-size:.85rem !important; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-eyebrow">Exploratory Data Analysis · Dataset 1</div>
    <div class="page-title">Amazon Products Overview</div>
    <div class="page-sub">วิเคราะห์ข้อมูล 1.4M สินค้า Amazon — ราคา, ดาว, รีวิว, และ Best Seller</div>
</div>
""", unsafe_allow_html=True)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ใช้ full dataset ถ้ามี ไม่งั้นใช้ sample (สำหรับ deploy บน cloud)
_full   = os.path.join(ROOT, 'data', 'amazon_products.csv')
_sample = os.path.join(ROOT, 'data', 'amazon_products_sample.csv')
DATA_PATH = _full if os.path.exists(_full) else _sample

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#64748b', size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', linecolor='rgba(255,255,255,0.06)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', linecolor='rgba(255,255,255,0.06)'),
)


@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path, encoding='utf-8')

    def clean_price(s):
        return pd.to_numeric(s.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

    if 'price'     in df.columns: df['price']     = clean_price(df['price'])
    if 'listPrice' in df.columns: df['listPrice'] = clean_price(df['listPrice'])
    if 'stars'     in df.columns: df['stars']     = pd.to_numeric(df['stars'],   errors='coerce')
    if 'reviews'   in df.columns: df['reviews']   = pd.to_numeric(df['reviews'], errors='coerce')

    df['isBestSeller'] = df['isBestSeller'].astype(str).str.lower().map(
        {'true':'Best Seller','false':'Not Best Seller','1':'Best Seller','0':'Not Best Seller'}
    ).fillna('Not Best Seller')

    df['discount_pct'] = np.where(
        (df.get('listPrice', pd.Series(dtype=float)) > 0) & (df.get('price', pd.Series(dtype=float)) > 0),
        ((df['listPrice'] - df['price']) / df['listPrice'] * 100).clip(0, 99), 0.0
    ) if 'listPrice' in df.columns else 0.0

    cat_col = next((c for c in ['main_category', 'category', 'category_id'] if c in df.columns), None)
    if cat_col: df['_category'] = df[cat_col].astype(str).str.strip()
    else:       df['_category'] = 'unknown'

    return df


if not os.path.exists(DATA_PATH):
    st.warning("⚠️ ไม่พบ data/amazon_products.csv หรือ amazon_products_sample.csv")
    st.code("kaggle datasets download -d asaniczka/amazon-products-dataset-2023-1-4m-products --unzip -p data/")
    st.stop()

if 'sample' in DATA_PATH:
    st.info("ℹ️ กำลังใช้ข้อมูล Sample 50,000 รายการ (full dataset 1.4M ต้องดาวน์โหลดเพิ่ม)")

with st.spinner("กำลังโหลดข้อมูล..."):
    df = load_data(DATA_PATH)

n_total      = len(df)
n_bs         = (df['isBestSeller'] == 'Best Seller').sum()
avg_price    = df['price'].dropna().median() if 'price' in df.columns else 0
avg_stars    = df['stars'].dropna().mean()   if 'stars' in df.columns else 0
avg_reviews  = df['reviews'].dropna().median() if 'reviews' in df.columns else 0
n_categories = df['_category'].nunique()

st.markdown(f"""
<div class="metric-row">
    <div class="metric-box"><div class="metric-val">{n_total/1e6:.1f}M</div><div class="metric-label">Total Products</div></div>
    <div class="metric-box"><div class="metric-val">{n_bs:,}</div><div class="metric-label">Best Sellers</div></div>
    <div class="metric-box"><div class="metric-val">{n_bs/n_total*100:.1f}%</div><div class="metric-label">Best Seller Rate</div></div>
    <div class="metric-box"><div class="metric-val">${avg_price:.0f}</div><div class="metric-label">Median Price</div></div>
    <div class="metric-box"><div class="metric-val">{avg_stars:.2f}⭐</div><div class="metric-label">Avg Stars</div></div>
    <div class="metric-box"><div class="metric-val">{int(avg_reviews):,}</div><div class="metric-label">Median Reviews</div></div>
    <div class="metric-box"><div class="metric-val">{n_categories:,}</div><div class="metric-label">Categories</div></div>
</div>
""", unsafe_allow_html=True)

# ── สุ่ม sample เพื่อ plot (ไม่ให้ช้า) ──────────────────
SAMPLE = 50_000
df_s = df.sample(min(SAMPLE, len(df)), random_state=42)

col1, col2 = st.columns(2, gap="medium")

# Best Seller Pie
with col1:
    st.markdown('<div class="panel"><div class="panel-title">Best Seller Distribution</div>', unsafe_allow_html=True)
    cnts = df['isBestSeller'].value_counts()
    fig = go.Figure(go.Pie(
        labels=cnts.index, values=cnts.values,
        hole=.55, marker_colors=['#f97316','#1e293b'],
        textfont=dict(size=13, color='white'),
        hovertemplate='%{label}: %{value:,} (%{percent})<extra></extra>'
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=True,
                      legend=dict(orientation='h', y=-.1, font=dict(color='#64748b')))
    st.plotly_chart(fig, width='stretch')
    st.markdown(f"""
    <div class="insight-box">
        <b>Insight:</b> มีเพียง <b>{n_bs/n_total*100:.1f}%</b> ของสินค้าทั้งหมดที่เป็น Best Seller
        ทำให้ dataset มีความ imbalance สูง — โมเดล ML ใช้ <b>scale_pos_weight</b> รับมือปัญหานี้
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Stars Distribution
with col2:
    st.markdown('<div class="panel"><div class="panel-title">Star Rating Distribution</div>', unsafe_allow_html=True)
    if 'stars' in df.columns:
        star_counts = df['stars'].dropna().round(1).value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=star_counts.index, y=star_counts.values,
            marker=dict(color='#f97316', opacity=0.8),
            hovertemplate='%{x} ดาว: %{y:,} สินค้า<extra></extra>'
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          xaxis_title='Stars', yaxis_title='จำนวนสินค้า')
        st.plotly_chart(fig, width='stretch')
        st.markdown(f"""
        <div class="insight-box">
            <b>Insight:</b> ดาวเฉลี่ย <b>{avg_stars:.2f}</b> — สินค้าส่วนใหญ่ได้คะแนนสูง (4–5 ดาว)
            สะท้อน selection bias ของ marketplace ที่สินค้าไม่ดีมักถูกลบออก
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="medium")

# Price Distribution
with col3:
    st.markdown('<div class="panel"><div class="panel-title">Price Distribution (< $200)</div>', unsafe_allow_html=True)
    if 'price' in df.columns:
        price_data = df_s['price'].dropna()
        price_data = price_data[(price_data > 0) & (price_data < 200)]
        fig = px.histogram(price_data, nbins=60,
                           color_discrete_sequence=['#f97316'])
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          xaxis_title='Price (USD)', yaxis_title='จำนวนสินค้า')
        fig.update_traces(opacity=0.8)
        st.plotly_chart(fig, width='stretch')
        st.markdown(f"""
        <div class="insight-box">
            <b>Insight:</b> ราคา median ที่ <b>${avg_price:.0f}</b> — สินค้าส่วนใหญ่อยู่ช่วง $10–$50
            การกระจายเป็น long-tail ขวา ต้องใช้ median แทน mean
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Stars vs Reviews (Best Seller vs Not)
with col4:
    st.markdown('<div class="panel"><div class="panel-title">Stars vs Reviews (Best Seller)</div>', unsafe_allow_html=True)
    if 'stars' in df.columns and 'reviews' in df.columns:
        plot_df = df_s.dropna(subset=['stars','reviews'])
        plot_df = plot_df[plot_df['reviews'] < plot_df['reviews'].quantile(0.95)]
        fig = px.scatter(
            plot_df.sample(min(5000, len(plot_df)), random_state=42),
            x='stars', y='reviews',
            color='isBestSeller',
            color_discrete_map={'Best Seller':'#f97316','Not Best Seller':'#1e3a5f'},
            opacity=0.6, size_max=5,
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          xaxis_title='Stars', yaxis_title='Reviews')
        st.plotly_chart(fig, width='stretch')
        st.markdown("""
        <div class="insight-box">
            <b>Insight:</b> Best Seller มักมี <b>ดาวสูง + รีวิวจำนวนมาก</b>
            ทั้งสองฟีเจอร์นี้เป็น signal สำคัญที่โมเดลใช้ทำนาย
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Top Categories
st.markdown('<div class="panel"><div class="panel-title">Best Sellers by Category (Top 15)</div>', unsafe_allow_html=True)
cat_group = df.groupby('_category')['isBestSeller'].apply(
    lambda x: (x == 'Best Seller').mean() * 100
).sort_values(ascending=False).head(15).reset_index()
cat_group.columns = ['category', 'best_seller_pct']

fig = px.bar(cat_group, x='best_seller_pct', y='category', orientation='h',
             color='best_seller_pct',
             color_continuous_scale=['#1e293b','#7c2d12','#f97316'],
             text=cat_group['best_seller_pct'].apply(lambda x: f'{x:.1f}%'))
fig.update_layout(**PLOTLY_LAYOUT, height=max(320, len(cat_group)*30),
                  xaxis_title='% Best Seller', yaxis_title='',
                  coloraxis_showscale=False)
fig.update_yaxes(categoryorder='total ascending')
fig.update_traces(textposition='outside', textfont_color='#64748b')
st.plotly_chart(fig, width='stretch')
st.markdown("</div>", unsafe_allow_html=True)

# Raw Data Preview
st.markdown('<div class="panel"><div class="panel-title">Raw Data Sample (500 แถว)</div>', unsafe_allow_html=True)
show_cols = [c for c in ['title','stars','reviews','price','listPrice',
                          'discount_pct','isBestSeller','_category'] if c in df.columns]
st.dataframe(
    df[show_cols].dropna(subset=['stars']).head(500),
    width='stretch',
    height=350
)
st.caption(f"แสดง 500 แถวแรกจาก {n_total:,} แถวทั้งหมด")
st.markdown("</div>", unsafe_allow_html=True)
