import streamlit as st
import os
import joblib

st.set_page_config(page_title="ML Model Info", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&family=Prompt:wght@500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #0F1117 !important;
    font-family: 'Sarabun', sans-serif;
    color: #2D3250;
}
[data-testid="stSidebar"] {
    background: #1A1D2E !important;
    border-right: 1px solid #2D3250 !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Sarabun', sans-serif !important;
}
section.main > div { padding: 2rem 2.5rem; }
[data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
/* ── SIDEBAR ─────────────────────────────── */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stSidebarContent"] { padding-top: 0 !important; }
[data-testid="stSidebarNavItems"]::before {
    content: '🛒 E-Commerce AI';
    display: block; font-family: 'Prompt', sans-serif;
    font-size: .9rem; font-weight: 700; color: #F0F4FF; letter-spacing: .01em;
    padding: 1.5rem 1.3rem 1rem; border-bottom: 1px solid #2D3250; margin-bottom: .5rem;
}
[data-testid="stSidebarNavItems"] a {
    display: flex !important; align-items: center !important;
    padding: .62rem 1rem !important; margin: .1rem .65rem !important;
    border-radius: 10px !important; font-family: 'Sarabun', sans-serif !important;
    font-size: .875rem !important; font-weight: 500 !important;
    color: #94A3B8 !important; text-decoration: none !important;
    transition: all .15s ease !important; border: 1px solid transparent !important;
}
[data-testid="stSidebarNavItems"] a:hover {
    background: rgba(255,107,53,0.08) !important; color: #E2E8F0 !important;
    border-color: rgba(255,107,53,0.18) !important; transform: translateX(3px) !important;
}
[data-testid="stSidebarNavItems"] a[aria-current="page"] {
    background: linear-gradient(90deg,rgba(255,107,53,.16),rgba(255,107,53,.04)) !important;
    color: #FF6B35 !important; border-color: rgba(255,107,53,.32) !important;
    font-weight: 700 !important; border-left: 3px solid #FF6B35 !important;
}
[data-testid="stSidebarNavItems"] a span,
[data-testid="stSidebarNavItems"] a p { color: inherit !important; }

.page-header {
    padding: 1.5rem 0 2rem;
    border-bottom: 1px solid #2D3250;
    margin-bottom: 2rem;
}
.page-eyebrow {
    font-family: 'Sarabun', sans-serif;
    font-size: .7rem; font-weight: 700; letter-spacing: .15em;
    text-transform: uppercase; color: #E85D04; margin-bottom: .5rem;
}
.page-title {
    font-family: 'Prompt', sans-serif;
    font-size: 2rem; font-weight: 700; color: #F0F4FF;
}
.page-sub {
    font-family: 'Sarabun', sans-serif;
    font-size: .95rem; color: #94A3B8; margin-top: .4rem; line-height: 1.6;
}

.metric-row {
    display: grid; grid-template-columns: repeat(auto-fit,minmax(140px,1fr));
    gap: 1rem; margin-bottom: 2rem;
}
.metric-box {
    background: #1A1D2E; border: 1px solid #2D3250;
    border-radius: 16px; padding: 1.3rem; text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.metric-val {
    font-family: 'Prompt', sans-serif;
    font-size: 1.7rem; font-weight: 700; color: #FF6B35;
}
.metric-label {
    font-family: 'Sarabun', sans-serif;
    font-size: .75rem; color: #64748B; margin-top: .2rem;
}

.panel {
    background: #1A1D2E; border: 1px solid #2D3250;
    border-top: 3px solid #FF6B35;
    border-radius: 16px; padding: 1.8rem; margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.panel.purple-top { border-top-color: #9D6FFF; }
.panel.blue-top   { border-top-color: #60A5FA; }
.panel.green-top  { border-top-color: #34D399; }

.panel-title {
    font-family: 'Prompt', sans-serif;
    font-size: .85rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #64748B;
    margin-bottom: 1.4rem; display: flex; align-items: center; gap: .5rem;
}
.panel-title::after { content: ''; flex: 1; height: 1px; background: #2D3250; }

/* Theory boxes */
.theory-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
.theory-card {
    background: #1E2235; border: 1px solid #2D3250;
    border-radius: 12px; padding: 1.2rem;
}
.theory-card-icon { font-size: 1.3rem; margin-bottom: .5rem; }
.theory-card-title {
    font-family: 'Prompt', sans-serif;
    font-size: .95rem; font-weight: 600; color: #F0F4FF; margin-bottom: .5rem;
}
.theory-card-badge {
    display: inline-block; padding: .2rem .6rem; border-radius: 6px;
    font-family: 'Sarabun', sans-serif;
    font-size: .68rem; font-weight: 600; margin-bottom: .5rem;
}
.badge-orange { background: rgba(255,107,53,0.15); color: #FF6B35; }
.badge-purple { background: rgba(157,111,255,0.15); color: #9D6FFF; }
.badge-blue   { background: rgba(96,165,250,0.15); color: #60A5FA; }
.badge-green  { background: rgba(52,211,153,0.15); color: #34D399; }
.theory-card-text {
    font-family: 'Sarabun', sans-serif;
    font-size: .86rem; color: #94A3B8; line-height: 1.7;
}

.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: .8rem; }
.feature-card {
    background: #1E2235; border: 1px solid #2D3250;
    border-radius: 12px; padding: .9rem 1rem;
}
.feature-name {
    font-family: 'Prompt', sans-serif;
    font-size: .85rem; font-weight: 600; color: #F0F4FF; margin-bottom: .2rem;
}
.feature-desc {
    font-family: 'Sarabun', sans-serif;
    font-size: .8rem; color: #94A3B8; line-height: 1.55;
}
.feature-explain {
    font-family: 'Sarabun', sans-serif;
    font-size: .78rem; color: #64748B; line-height: 1.5;
    margin-top: .3rem; font-style: italic;
}
.feature-tag {
    display: inline-block; margin-top: .4rem;
    font-family: 'Sarabun', sans-serif;
    font-size: .65rem; font-weight: 600; padding: .2rem .5rem; border-radius: 6px;
}
.tag-num { background: rgba(255,107,53,0.15); color: #FF6B35; }
.tag-eng { background: rgba(96,165,250,0.15); color: #60A5FA; }
.tag-cat { background: rgba(157,111,255,0.15); color: #9D6FFF; }

.arch-row { display: flex; align-items: stretch; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
.arch-card {
    flex: 1; min-width: 160px; background: #1E2235;
    border: 1px solid #2D3250;
    border-radius: 12px; padding: 1.2rem; text-align: center;
}
.arch-icon { font-size: 1.5rem; margin-bottom: .5rem; }
.arch-title {
    font-family: 'Prompt', sans-serif;
    font-size: .88rem; font-weight: 600; color: #F0F4FF; margin-bottom: .3rem;
}
.arch-sub {
    font-family: 'Sarabun', sans-serif;
    font-size: .78rem; color: #94A3B8; line-height: 1.55;
}
.arch-badge {
    display: inline-block; margin-top: .5rem;
    font-family: 'Sarabun', sans-serif;
    font-size: .65rem; font-weight: 600; padding: .22rem .5rem;
    border-radius: 6px; background: rgba(255,107,53,0.15); color: #FF6B35;
}

.flow-step {
    display: flex; align-items: flex-start; gap: 1rem; padding: .8rem 1rem;
    background: #1E2235; border-radius: 10px; margin-bottom: .5rem;
    border: 1px solid #2D3250;
}
.flow-num {
    width: 28px; height: 28px; border-radius: 50%;
    background: rgba(255,107,53,0.15); color: #FF6B35;
    font-size: .75rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    font-family: 'Prompt', sans-serif;
}
.flow-text {
    font-family: 'Sarabun', sans-serif;
    font-size: .86rem; color: #94A3B8; line-height: 1.6;
}
.flow-text b { color: #F0F4FF; }

.not-trained {
    background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.25);
    border-radius: 12px; padding: 1rem 1.4rem;
    font-family: 'Sarabun', sans-serif;
}
.not-trained b { color: #FBBF24; }

.imbalance-box {
    background: #1E2235; border: 1px solid #2D3250;
    border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
    font-family: 'Sarabun', sans-serif; font-size: .88rem; color: #94A3B8; line-height: 1.8;
}
.imbalance-box b { color: #F0F4FF; }

.ensemble-result {
    text-align: center; border: 2px solid rgba(255,107,53,0.3);
    background: rgba(255,107,53,0.06);
    border-radius: 12px; padding: 1rem;
    font-family: 'Sarabun', sans-serif;
}
.ensemble-result-title {
    font-family: 'Prompt', sans-serif;
    font-size: 1rem; font-weight: 600; color: #FF6B35;
}
.ensemble-result-sub {
    font-size: .82rem; color: #94A3B8; margin-top: .3rem; line-height: 1.6;
}

label { color: #64748B !important; font-family: 'Sarabun', sans-serif !important; font-size: .88rem !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
<div style="position:fixed;bottom:0;left:0;width:241px;padding:1rem 1.3rem;border-top:1px solid #2D3250;background:#1A1D2E">
  <div style="font-family:'Sarabun',sans-serif;font-size:.72rem;color:#475569;line-height:1.9">
    🎓 AI Project · 2024<br>
    <span style="color:#3D4766">MobileNetV2 · Ensemble ML</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-eyebrow">Machine Learning · Structured Data</div>
    <div class="page-title">ML Model — Sales Volume Predictor</div>
    <div class="page-sub">Ensemble Classifier: RF + XGBoost + HistGBM ทำนายระดับยอดขายต่อเดือน → Low / Medium / High<br>ใช้ข้อมูลตัวเลข 9 features จากราคา, ดาว, รีวิว และ category ของสินค้า</div>
</div>
""", unsafe_allow_html=True)

ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
meta_path = os.path.join(ROOT, 'models', 'sales_model_meta.pkl')

# Load meta
if os.path.exists(meta_path):
    meta    = joblib.load(meta_path)
    acc     = meta.get('accuracy', 0) * 100
    f1      = meta.get('f1_weighted', 0)
    f1_mac  = meta.get('f1_macro', 0)
    n_train = meta.get('n_train', 0)
    trained = True
else:
    meta    = {}
    trained = False

if trained:
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box"><div class="metric-val">{acc:.1f}%</div><div class="metric-label">Accuracy</div></div>
        <div class="metric-box"><div class="metric-val">{f1:.4f}</div><div class="metric-label">F1 Weighted</div></div>
        <div class="metric-box"><div class="metric-val">{f1_mac:.4f}</div><div class="metric-label">F1 Macro</div></div>
        <div class="metric-box"><div class="metric-val">{n_train:,}</div><div class="metric-label">Training Samples</div></div>
        <div class="metric-box"><div class="metric-val">3</div><div class="metric-label">Ensemble Models</div></div>
        <div class="metric-box"><div class="metric-val">{meta.get('xgb_device','cpu').upper()}</div><div class="metric-label">XGB Device</div></div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="metric-row">
        <div class="metric-box"><div class="metric-val">—</div><div class="metric-label">Accuracy</div></div>
        <div class="metric-box"><div class="metric-val">—</div><div class="metric-label">F1 Weighted</div></div>
        <div class="metric-box"><div class="metric-val">—</div><div class="metric-label">F1 Macro</div></div>
        <div class="metric-box"><div class="metric-val">500K</div><div class="metric-label">Max Samples</div></div>
        <div class="metric-box"><div class="metric-val">3</div><div class="metric-label">Ensemble Models</div></div>
        <div class="metric-box"><div class="metric-val">GPU/CPU</div><div class="metric-label">XGB Device</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="not-trained">
        ⚠️ <b>ยังไม่ได้เทรนโมเดล</b> — รัน <code>python train_sales_model.py</code> เพื่อเทรน
    </div>
    <br>
    """, unsafe_allow_html=True)

# ── THEORY SECTION ────────────────────────────────────────
st.markdown('<div class="panel"><div class="panel-title">ทฤษฎีโมเดล Machine Learning</div>', unsafe_allow_html=True)
st.markdown("""
<div class="theory-grid">
    <div class="theory-card">
        <div class="theory-card-icon">🌳</div>
        <div class="theory-card-title">Random Forest</div>
        <span class="theory-card-badge badge-orange">Bagging Ensemble</span>
        <div class="theory-card-text">
            Random Forest สร้าง Decision Tree จำนวนมาก (200 ต้น) โดยแต่ละต้น
            เทรนบน <strong>subset ของข้อมูลและ features ที่สุ่มมา</strong>
            แล้วนำผลโหวตมารวมกัน (majority vote)<br><br>
            จุดเด่น: ต้านทาน noise ได้ดี, ไม่ค่อย overfit,
            รองรับ class imbalance ด้วย <code>class_weight='balanced'</code>
            ที่ปรับน้ำหนัก sample ของ minority class ให้สูงขึ้นอัตโนมัติ
        </div>
    </div>
    <div class="theory-card">
        <div class="theory-card-icon">🚀</div>
        <div class="theory-card-title">XGBoost</div>
        <span class="theory-card-badge badge-blue">Gradient Boosting</span>
        <div class="theory-card-text">
            XGBoost สร้าง tree แบบ <strong>sequential</strong> โดยแต่ละต้นพยายาม
            แก้ error ของต้นก่อนหน้า ใช้ gradient descent ใน function space<br><br>
            จุดเด่น: มักให้ accuracy สูงที่สุดในข้อมูล tabular, รองรับ GPU,
            มี L1+L2 regularization ป้องกัน overfitting,
            ใช้ <code>objective='multi:softprob'</code> เพื่อ output probability ทั้ง 3 class
        </div>
    </div>
    <div class="theory-card">
        <div class="theory-card-icon">📈</div>
        <div class="theory-card-title">HistGradientBoosting</div>
        <span class="theory-card-badge badge-purple">Fast Boosting</span>
        <div class="theory-card-text">
            HistGBM ของ scikit-learn ใช้ <strong>histogram-based split</strong>
            คล้าย LightGBM ทำให้เร็วมากและใช้ RAM น้อย เหมาะกับข้อมูลขนาดใหญ่<br><br>
            จุดเด่น: รองรับ missing values ในตัว, มี EarlyStopping หยุดเทรนเมื่อ
            validation loss ไม่ดีขึ้น, ใช้ <code>class_weight='balanced'</code>
            เหมือน Random Forest
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# VotingClassifier explanation
st.markdown("""
<div style="margin: 1.2rem 0 .5rem; text-align:center; font-family:'Sarabun',sans-serif; font-size:.9rem; color:#718096;">
    ↓ รวมกันด้วย VotingClassifier (Soft Voting) ↓
</div>
<div class="ensemble-result">
    <div class="ensemble-result-title">VotingClassifier — Soft Voting</div>
    <div class="ensemble-result-sub">
        แต่ละโมเดลส่งออก <strong>probability ของแต่ละ class</strong> (Low / Medium / High)<br>
        จากนั้นนำค่าเฉลี่ยของ probability มาเปรียบเทียบ → เลือก class ที่มี probability สูงสุด<br>
        วิธีนี้ดีกว่า Hard Voting เพราะใช้ "ความมั่นใจ" ของโมเดล ไม่ใช่แค่การโหวต
    </div>
</div>

<div style="margin-top:1.2rem; font-family:'Sarabun',sans-serif; font-size:.9rem; color:#4A5568; line-height:1.8; padding: 1rem; background:#F7FAFC; border-radius:12px; border:1px solid #2D3250;">
    <strong style="color:#E2E8F0; font-family:'Prompt',sans-serif;">ทำไมถึงเลือก 3 โมเดลนี้?</strong><br>
    เลือกโมเดลที่มี <strong>วิธีคิดต่างกัน</strong> เพื่อให้ความผิดพลาดของแต่ละโมเดลไม่เหมือนกัน
    (Bagging vs Boosting) Random Forest แก้จุดอ่อนของ Boosting ที่ sensitive ต่อ outlier,
    XGBoost ให้ accuracy สูง, HistGBM ประมวลผลเร็วและเสถียร การรวม 3 โมเดลทำให้
    <strong>variance ลดลง</strong> และ generalization ดีขึ้น
</div>
</div>
""", unsafe_allow_html=True)

# Features
st.markdown('<div class="panel"><div class="panel-title">Features ที่ใช้ทำนาย (9 ตัว + 1 Categorical)</div>', unsafe_allow_html=True)

features = [
    ("stars",            "คะแนนดาว 0–5",
     "ค่าเฉลี่ยคะแนนที่ลูกค้าให้สินค้า ยิ่งสูงยิ่งดี สะท้อนความพึงพอใจของผู้ซื้อ",
     "Numeric", "tag-num"),
    ("reviews",          "จำนวนรีวิวทั้งหมด",
     "ยิ่งรีวิวมาก ยิ่งแสดงว่ามีคนซื้อและใช้สินค้าจริง เป็นสัญญาณความน่าเชื่อถือ",
     "Numeric", "tag-num"),
    ("price_thb",        "ราคาขายจริง (บาท) = price × 35",
     "ราคาจริงของสินค้าแปลงจาก USD เป็นบาท (×35) ราคาต่ำกว่าอาจขายดีกว่า",
     "Numeric", "tag-num"),
    ("listPrice_thb",    "ราคาตั้งต้น (บาท) = listPrice × 35",
     "ราคาก่อนลด ใช้คำนวณ discount_pct ถ้าไม่มีการลดราคาจะเท่ากับ price_thb",
     "Numeric", "tag-num"),
    ("discount_pct",     "(listPrice−price)/listPrice × 100",
     "% ส่วนลด สินค้าที่ลดราคามากมักดึงดูดผู้ซื้อได้ดีกว่า",
     "Engineered", "tag-eng"),
    ("log_reviews",      "log1p(reviews) — ลด right-skew",
     "แปลง reviews ด้วย log เพราะสินค้าส่วนใหญ่มีรีวิวน้อย แต่บางตัวมีแสน ทำให้ distribution สมมาตรขึ้น",
     "Engineered", "tag-eng"),
    ("price_per_review", "price_thb / reviews",
     "ราคาต่อ 1 รีวิว ยิ่งต่ำ แสดงว่ามีคนรีวิวเยอะเมื่อเทียบกับราคา",
     "Engineered", "tag-eng"),
    ("value_score",      "stars × log_reviews",
     "คะแนนรวม = ดาว × ปริมาณรีวิว สินค้าดีและมีคนรีวิวเยอะจะได้คะแนนสูง",
     "Engineered", "tag-eng"),
    ("review_density",   "reviews / price_thb",
     "ความหนาแน่นรีวิวต่อบาท สินค้าที่ราคาถูกและมีรีวิวเยอะจะได้คะแนนสูง",
     "Engineered", "tag-eng"),
    ("cat_name",         "ชื่อหมวดหมู่สินค้า (OneHotEncode)",
     "หมวดหมู่สินค้า เช่น Electronics, Clothing แต่ละ category มีพฤติกรรมยอดขายต่างกัน",
     "Categorical", "tag-cat"),
]
st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
for name, desc, explain, ftype, tag_cls in features:
    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-name">{name}</div>
        <div class="feature-desc">{desc}</div>
        <div class="feature-explain">{explain}</div>
        <span class="feature-tag {tag_cls}">{ftype}</span>
    </div>
    """, unsafe_allow_html=True)
st.markdown("</div></div>", unsafe_allow_html=True)

# Architecture
st.markdown('<div class="panel purple-top"><div class="panel-title">Ensemble Architecture</div>', unsafe_allow_html=True)
st.markdown('<div class="arch-row">', unsafe_allow_html=True)
for icon, title, sub, badge in [
    ("🌳", "Random Forest", "200 trees · class_weight=balanced · n_jobs=-1 (ทุก CPU core)", "Bagging"),
    ("🚀", "XGBoost", "300 trees · scale_pos_weight · GPU/CPU hist · L1+L2 regularization", "Boosting"),
    ("📈", "HistGradientBoosting", "300 iterations · class_weight=balanced · EarlyStopping", "Boosting"),
]:
    st.markdown(f"""
    <div class="arch-card">
        <div class="arch-icon">{icon}</div>
        <div class="arch-title">{title}</div>
        <div class="arch-sub">{sub}</div>
        <span class="arch-badge">{badge}</span>
    </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-family:'Sarabun',sans-serif; color:#718096; margin:.5rem 0; font-size:.88rem">
    ↓ VotingClassifier (soft voting — average probability จากทั้ง 3 โมเดล) ↓
</div>
<div class="arch-card" style="text-align:center; border-color:rgba(232,93,4,0.25); background:rgba(232,93,4,0.04)">
    <div class="arch-icon">🏆</div>
    <div class="arch-title" style="color:#E85D04">Ensemble → Low / Medium / High</div>
    <div class="arch-sub">predict_proba → argmax → ระดับยอดขาย (Low ≤50, Medium 51–500, High &gt;500 ซื้อ/เดือน)</div>
</div>
</div>
""", unsafe_allow_html=True)

# Pipeline steps
st.markdown('<div class="panel blue-top"><div class="panel-title">Training Pipeline</div>', unsafe_allow_html=True)
steps = [
    ("1",  "<b>Load &amp; Merge</b> — อ่าน amazon_products.csv + amazon_categories.csv → merge ด้วย category_id"),
    ("2",  "<b>Clean</b> — แปลง price/stars/reviews เป็นตัวเลข, แปลงราคาจาก USD → บาท (×35)"),
    ("3",  "<b>Labeling</b> — แบ่ง boughtInLastMonth เป็น 3 ระดับ: Low(≤50), Medium(51–500), High(&gt;500)"),
    ("4",  "<b>Feature Engineering</b> — สร้าง discount_pct, log_reviews, value_score, review_density, price_per_review"),
    ("5",  "<b>Filter</b> — กรอง price_thb 1–350,000 บาท, stars 0–5"),
    ("6",  "<b>Sample</b> — ถ้าข้อมูล &gt; 500K แถว ทำ stratified sampling รักษาสัดส่วน class"),
    ("7",  "<b>Preprocessing</b> — StandardScaler + SimpleImputer(median) สำหรับ numeric,<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;OneHotEncoder(handle_unknown=ignore) สำหรับ category"),
    ("8",  "<b>Class Imbalance</b> — RF+HGB: class_weight='balanced', XGB: objective=multi:softprob"),
    ("9",  "<b>Train Ensemble</b> — VotingClassifier(soft) เทรน RF+XGB+HGB พร้อมกัน (n_jobs=-1)"),
    ("10", "<b>Evaluate</b> — Accuracy, F1 (weighted), F1 (macro) บน test set 20%"),
    ("11", "<b>Save</b> — sales_model.pkl + sales_model_meta.pkl + sales_category_map.pkl"),
]
for num, text in steps:
    st.markdown(f'<div class="flow-step"><div class="flow-num">{num}</div><div class="flow-text">{text}</div></div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Imbalance handling
st.markdown('<div class="panel green-top"><div class="panel-title">ปัญหา Class Imbalance และการแก้ไข</div>', unsafe_allow_html=True)
st.markdown("""
<div class="imbalance-box">
    <strong style="font-family:'Prompt',sans-serif; font-size:.95rem;">Class Imbalance คืออะไร?</strong><br>
    ในชีวิตจริง สินค้าส่วนใหญ่ขายได้ <strong>น้อย (Low)</strong> มีเพียงส่วนน้อยที่ขายดีมาก (High)
    Dataset นี้มีสัดส่วนประมาณ: <strong>Low 75.4%</strong> / <strong>Medium 18.9%</strong> / <strong>High 5.7%</strong><br><br>
    ถ้าไม่จัดการ โมเดลจะ "ขี้เกียจ" — ทำนาย Low ทุกตัวก็ได้ accuracy 75% แต่ไม่มีประโยชน์จริง
</div>
<div class="imbalance-box">
    <strong style="font-family:'Prompt',sans-serif; font-size:.95rem;">วิธีแก้ที่ใช้ในโปรเจกต์นี้</strong><br>
    • <b>class_weight='balanced'</b> ใน Random Forest และ HistGBM<br>
    &nbsp;&nbsp;→ ปรับน้ำหนัก sample ของ minority class (High) ให้สูงขึ้น โมเดลจะ "สนใจ" class เล็กมากขึ้น<br><br>
    • <b>objective='multi:softprob'</b> ใน XGBoost<br>
    &nbsp;&nbsp;→ output probability ทั้ง 3 class ทำให้ Soft Voting ทำงานได้ถูกต้อง<br><br>
    • <b>VotingClassifier(soft)</b> — average probability จาก 3 โมเดล ลด variance และ bias<br><br>
    • ใช้ <b>F1 (macro)</b> เป็น metric หลัก — ให้ความสำคัญกับ High class เท่ากับ Low class<br>
    • ใช้ <b>F1 (weighted)</b> เป็น metric รอง — ถ่วงน้ำหนักตามสัดส่วน class จริง
</div>
</div>
""", unsafe_allow_html=True)

st.code("python train_sales_model.py", language="bash")

# References
st.markdown("""
<div class="panel">
<div class="panel-title">แหล่งอ้างอิง (References)</div>
<div class="flow-text" style="line-height:2.2">
<b style="color:#E2E8F0; font-family:'Prompt',sans-serif;">Dataset</b><br>
• Sazickka, A. (2023). <i>Amazon Products Dataset 2023</i>. Kaggle.
  <span style="color:#475569">— https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023</span><br>
<br>
<b style="color:#E2E8F0; font-family:'Prompt',sans-serif;">Algorithms</b><br>
• Breiman, L. (2001). Random Forests. <i>Machine Learning</i>, 45(1), 5–32.
  doi:10.1023/A:1010933404324<br>
• Chen, T., &amp; Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
  <i>Proceedings of KDD 2016</i>. doi:10.1145/2939672.2939785<br>
• Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
  <i>The Annals of Statistics</i>, 29(5), 1189–1232.<br>
• Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
  <i>JAIR</i>, 16, 321–357.<br>
<br>
<b style="color:#E2E8F0; font-family:'Prompt',sans-serif;">Libraries</b><br>
• Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. <i>JMLR</i>, 12, 2825–2830.<br>
• XGBoost Documentation. — https://xgboost.readthedocs.io<br>
• Lemaître, G., et al. (2017). Imbalanced-learn. <i>JMLR</i>, 18(17), 1–5.
</div>
</div>
""", unsafe_allow_html=True)
