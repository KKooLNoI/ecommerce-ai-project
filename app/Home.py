import streamlit as st

st.set_page_config(
    page_title="E-Commerce Product Review AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&family=Prompt:wght@500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] {
    background: #F0F4F8 !important;
    font-family: 'Sarabun', sans-serif;
    color: #2D3748;
}
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Sarabun', sans-serif !important;
    color: #2D3748 !important;
}
section.main > div { padding: 0 !important; }
[data-testid="stDecoration"], header { display: none !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #FFFFFF 0%, #F7F9FC 50%, #EEF2FF 100%);
    padding: 4rem 3rem 3.5rem;
    position: relative; overflow: hidden;
    border-bottom: 1px solid #E2E8F0;
}
.hero::before {
    content: '';
    position: absolute; top: -30%; left: -10%;
    width: 50%; height: 150%;
    background: radial-gradient(ellipse, rgba(232,93,4,0.06) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute; bottom: -20%; right: -5%;
    width: 40%; height: 120%;
    background: radial-gradient(ellipse, rgba(124,58,237,0.05) 0%, transparent 65%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: .5rem;
    font-family: 'Sarabun', sans-serif;
    font-size: .72rem; font-weight: 700; letter-spacing: .15em; text-transform: uppercase;
    color: #E85D04; background: rgba(232,93,4,0.08);
    border: 1px solid rgba(232,93,4,0.2);
    border-radius: 999px; padding: .35rem .9rem; margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Prompt', sans-serif;
    font-size: clamp(2rem, 4.5vw, 3.4rem);
    font-weight: 700; color: #1A202C; line-height: 1.2;
    margin-bottom: 1rem;
}
.hero-title span { color: #E85D04; }
.hero-sub {
    font-family: 'Sarabun', sans-serif;
    font-size: 1.05rem; color: #4A5568; max-width: 600px;
    line-height: 1.8; margin-bottom: 2rem;
}
.hero-pills { display: flex; flex-wrap: wrap; gap: .6rem; }
.pill {
    font-family: 'Sarabun', sans-serif;
    font-size: .75rem; font-weight: 600; letter-spacing: .05em;
    padding: .4rem 1rem;
    border-radius: 999px; border: 1px solid;
}
.pill-orange { color: #E85D04; background: rgba(232,93,4,0.07); border-color: rgba(232,93,4,0.2); }
.pill-purple { color: #7C3AED; background: rgba(124,58,237,0.07); border-color: rgba(124,58,237,0.2); }
.pill-blue   { color: #2563EB; background: rgba(37,99,235,0.07);  border-color: rgba(37,99,235,0.2); }
.pill-green  { color: #059669; background: rgba(5,150,105,0.07);  border-color: rgba(5,150,105,0.2); }

/* Content area */
.content { padding: 3rem 3rem; }

.section-header { margin-bottom: 1.8rem; }
.section-eyebrow {
    font-family: 'Sarabun', sans-serif;
    font-size: .7rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #718096; margin-bottom: .4rem;
}
.section-title {
    font-family: 'Prompt', sans-serif;
    font-size: 1.5rem; font-weight: 600; color: #1A202C;
}

/* Nav cards */
.nav-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px,1fr)); gap: 1.2rem; }
.nav-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #E85D04;
    border-radius: 16px; padding: 1.6rem;
    text-decoration: none; display: block;
    transition: transform .2s, box-shadow .2s;
    cursor: default;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.nav-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.10); }
.nav-card.purple-top { border-top-color: #7C3AED; }
.nav-card.blue-top   { border-top-color: #2563EB; }
.nav-card.green-top  { border-top-color: #059669; }
.nav-icon { font-size: 1.7rem; margin-bottom: .9rem; }
.nav-num {
    font-family: 'Sarabun', sans-serif;
    font-size: .65rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #A0AEC0; margin-bottom: .3rem;
}
.nav-title {
    font-family: 'Prompt', sans-serif;
    font-size: 1rem; font-weight: 600; color: #1A202C; margin-bottom: .4rem;
}
.nav-desc  {
    font-family: 'Sarabun', sans-serif;
    font-size: .88rem; color: #4A5568; line-height: 1.65;
}
.nav-tag {
    display: inline-block; margin-top: .8rem;
    font-family: 'Sarabun', sans-serif;
    font-size: .68rem; font-weight: 600; letter-spacing: .05em;
    text-transform: uppercase; padding: .28rem .65rem;
    border-radius: 8px;
}
.tag-ml     { background: rgba(232,93,4,0.10);   color: #E85D04; }
.tag-cnn    { background: rgba(124,58,237,0.10);  color: #7C3AED; }
.tag-eda    { background: rgba(37,99,235,0.10);   color: #2563EB; }
.tag-result { background: rgba(5,150,105,0.10);   color: #059669; }

/* Explanation cards */
.explain-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); gap: 1rem; margin-top: 1.5rem; }
.explain-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px; padding: 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.explain-card-icon { font-size: 1.4rem; margin-bottom: .6rem; }
.explain-card-title {
    font-family: 'Prompt', sans-serif;
    font-size: .95rem; font-weight: 600; color: #1A202C; margin-bottom: .5rem;
}
.explain-card-text {
    font-family: 'Sarabun', sans-serif;
    font-size: .88rem; color: #4A5568; line-height: 1.7;
}

/* Pipeline diagram */
.pipeline {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px; padding: 2rem; margin-top: 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.pipe-steps { display: flex; align-items: center; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }
.pipe-step {
    background: #F7FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px; padding: .75rem 1.1rem;
    text-align: center; min-width: 110px;
}
.pipe-step-icon { font-size: 1.1rem; margin-bottom: .25rem; }
.pipe-step-label {
    font-family: 'Sarabun', sans-serif;
    font-size: .72rem; font-weight: 600; color: #718096;
}
.pipe-arrow { color: #CBD5E0; font-size: 1.2rem; }

/* Tier explanation */
.tier-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin-top: 1.5rem; }
.tier-card {
    background: #FFFFFF; border-radius: 16px; padding: 1.3rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    text-align: center;
}
.tier-badge {
    display: inline-block; padding: .3rem .9rem;
    border-radius: 999px; font-family: 'Prompt', sans-serif;
    font-size: .9rem; font-weight: 600; margin-bottom: .6rem;
}
.tier-high { background: rgba(5,150,105,0.1); color: #059669; }
.tier-med  { background: rgba(234,179,8,0.1);  color: #B45309; }
.tier-low  { background: rgba(239,68,68,0.1);  color: #DC2626; }
.tier-range {
    font-family: 'Sarabun', sans-serif;
    font-size: .82rem; font-weight: 600; color: #2D3748; margin-bottom: .4rem;
}
.tier-desc {
    font-family: 'Sarabun', sans-serif;
    font-size: .82rem; color: #4A5568; line-height: 1.6;
}

/* Stats */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap: 1rem; margin-top: 2rem; }
.stat-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px; padding: 1.2rem; text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.stat-val {
    font-family: 'Prompt', sans-serif;
    font-size: 1.6rem; font-weight: 700; color: #E85D04;
}
.stat-label {
    font-family: 'Sarabun', sans-serif;
    font-size: .75rem; color: #718096; margin-top: .2rem;
}

label { color: #718096 !important; font-family: 'Sarabun', sans-serif !important; font-size: .88rem !important; }
</style>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🛒 E-Commerce · AI System</div>
    <div class="hero-title">
        Product Review AI<br>
        <span>วิเคราะห์สินค้า 2 มุมมอง</span>
    </div>
    <div class="hero-sub">
        ระบบ AI วิเคราะห์สินค้า Amazon จาก 2 แหล่งข้อมูล:<br>
        <strong>ML Ensemble</strong> ทำนายความนิยมจากตัวเลข (ราคา, ดาว, รีวิว) ·
        <strong>CNN</strong> วิเคราะห์หมวดหมู่จากรูปภาพสินค้า<br>
        รวมผลเป็น Buy Score เพื่อแนะนำว่าสินค้าตัวไหน "น่าซื้อ"
    </div>
    <div class="hero-pills">
        <span class="pill pill-orange">Amazon Products 1.4M</span>
        <span class="pill pill-orange">XGBoost + Random Forest + HistGBM</span>
        <span class="pill pill-purple">MobileNetV2</span>
        <span class="pill pill-blue">Ensemble Learning</span>
        <span class="pill pill-green">Buy Score</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="content">', unsafe_allow_html=True)

# ML ENSEMBLE EXPLANATION
st.markdown("""
<div class="section-header" style="margin-top:0">
    <div class="section-eyebrow">ระบบ AI</div>
    <div class="section-title">ระบบนี้ทำงานอย่างไร?</div>
</div>
<div class="explain-grid">
    <div class="explain-card">
        <div class="explain-card-icon">🤖</div>
        <div class="explain-card-title">ML Ensemble คืออะไร?</div>
        <div class="explain-card-text">
            Ensemble Learning คือการนำโมเดล ML หลายตัวมาทำงานร่วมกัน แทนที่จะพึ่งโมเดลเดียว
            ระบบนี้ใช้ 3 โมเดล (Random Forest + XGBoost + HistGBM) โดยให้แต่ละโมเดลทำนาย
            แล้วนำ <strong>ค่าเฉลี่ย probability</strong> มาตัดสินใจ (Soft Voting)
            ผลลัพธ์มีความแม่นยำสูงกว่าโมเดลเดี่ยว และลด overfitting
        </div>
    </div>
    <div class="explain-card">
        <div class="explain-card-icon">🔢</div>
        <div class="explain-card-title">ทำไมต้องใช้ 3 โมเดล?</div>
        <div class="explain-card-text">
            แต่ละโมเดลมีจุดแข็งต่างกัน — <strong>Random Forest</strong> ต้านทาน noise ได้ดี,
            <strong>XGBoost</strong> จัดการ imbalanced data ได้เก่ง,
            <strong>HistGBM</strong> เร็วและใช้หน่วยความจำน้อย
            การรวม 3 โมเดลช่วย "ถ่วงดุล" จุดอ่อนของแต่ละตัว ทำให้ผลรวมดีกว่า
        </div>
    </div>
    <div class="explain-card">
        <div class="explain-card-icon">🧠</div>
        <div class="explain-card-title">CNN / Neural Network ทำอะไร?</div>
        <div class="explain-card-text">
            Convolutional Neural Network (CNN) วิเคราะห์รูปภาพสินค้า โดยเรียนรู้ pattern
            เช่น สี รูปทรง และ texture จากรูป ระบบใช้ <strong>MobileNetV2</strong>
            ที่เทรนด้วย ImageNet มาแล้ว (Transfer Learning) แล้ว fine-tune เพิ่มเติม
            ให้จำแนก category สินค้า Amazon ได้อย่างแม่นยำ
        </div>
    </div>
    <div class="explain-card">
        <div class="explain-card-icon">📦</div>
        <div class="explain-card-title">Dataset ที่ใช้คืออะไร?</div>
        <div class="explain-card-text">
            ใช้ <strong>Amazon Products Dataset 2023</strong> จาก Kaggle
            มีสินค้ากว่า 1.4 ล้านรายการ ครอบคลุมหลายหมวดหมู่
            ข้อมูลมีทั้งราคา, คะแนนดาว, จำนวนรีวิว, ราคาตั้งต้น
            และจำนวนซื้อในเดือนที่แล้ว (boughtInLastMonth) ที่ใช้เป็น label
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# TIER EXPLANATION
st.markdown("""
<div class="section-header" style="margin-top:2.5rem">
    <div class="section-eyebrow">ผลการทำนาย</div>
    <div class="section-title">ระดับยอดขาย Low / Medium / High หมายความว่าอย่างไร?</div>
</div>
<div class="tier-grid">
    <div class="tier-card">
        <div class="tier-badge tier-high">High</div>
        <div class="tier-range">มากกว่า 500 ซื้อ/เดือน</div>
        <div class="tier-desc">สินค้าขายดีมาก มีฐานลูกค้ากว้าง ความต้องการในตลาดสูง เหมาะสำหรับการลงทุนและขยาย stock</div>
    </div>
    <div class="tier-card">
        <div class="tier-badge tier-med">Medium</div>
        <div class="tier-range">51 – 500 ซื้อ/เดือน</div>
        <div class="tier-desc">ยอดขายปานกลาง สินค้ามีฐานลูกค้า มีโอกาสเติบโตถ้าปรับกลยุทธ์ราคาหรือการตลาด</div>
    </div>
    <div class="tier-card">
        <div class="tier-badge tier-low">Low</div>
        <div class="tier-range">0 – 50 ซื้อ/เดือน</div>
        <div class="tier-desc">ยอดขายต่ำ อาจเป็น niche product หรือสินค้ายังไม่เป็นที่รู้จัก ควรพิจารณาปรับราคาหรือ marketing</div>
    </div>
</div>
""", unsafe_allow_html=True)

# NAV CARDS
st.markdown("""
<div class="section-header" style="margin-top:2.5rem">
    <div class="section-eyebrow">Navigation</div>
    <div class="section-title">เลือกหัวข้อที่ต้องการสำรวจ</div>
</div>
<div class="nav-grid">
    <div class="nav-card blue-top">
        <div class="nav-icon">📊</div>
        <div class="nav-num">Page 01</div>
        <div class="nav-title">Data Overview</div>
        <div class="nav-desc">EDA ข้อมูล Amazon Products 1.4M รายการ — การกระจายของราคา, ดาว, รีวิว และ best seller พร้อม chart แบบ interactive</div>
        <span class="nav-tag tag-eda">EDA · Visualization</span>
    </div>
    <div class="nav-card">
        <div class="nav-icon">🤖</div>
        <div class="nav-num">Page 02</div>
        <div class="nav-title">ML Model Info</div>
        <div class="nav-desc">อธิบาย Ensemble Classifier: RF + XGBoost + HistGBM — ทฤษฎีแต่ละโมเดล, pipeline การเทรน, การจัดการ class imbalance</div>
        <span class="nav-tag tag-ml">Structured Data · ทฤษฎี</span>
    </div>
    <div class="nav-card purple-top">
        <div class="nav-icon">🧠</div>
        <div class="nav-num">Page 03</div>
        <div class="nav-title">CNN Model Info</div>
        <div class="nav-desc">อธิบาย MobileNetV2 Transfer Learning 2-Phase — สถาปัตยกรรม CNN, data augmentation, และเหตุผลที่เลือก architecture นี้</div>
        <span class="nav-tag tag-cnn">Image Classification · ทฤษฎี</span>
    </div>
    <div class="nav-card">
        <div class="nav-icon">🔢</div>
        <div class="nav-num">Page 05</div>
        <div class="nav-title">ทดสอบ ML Model</div>
        <div class="nav-desc">ใส่ราคา, คะแนนดาว, จำนวนรีวิว → Ensemble ทำนายว่าสินค้านี้อยู่ระดับ Low / Medium / High พร้อม probability แต่ละระดับ</div>
        <span class="nav-tag tag-ml">ML · Demo</span>
    </div>
    <div class="nav-card purple-top">
        <div class="nav-icon">🖼️</div>
        <div class="nav-num">Page 06</div>
        <div class="nav-title">ทดสอบ CNN Model</div>
        <div class="nav-desc">อัปโหลดรูปสินค้า → CNN จำแนกหมวดหมู่ด้วย MobileNetV2 พร้อมแสดง Top-5 predictions และ confidence score</div>
        <span class="nav-tag tag-cnn">CNN · Demo</span>
    </div>
</div>
""", unsafe_allow_html=True)

# PIPELINE
st.markdown("""
<div class="pipeline">
    <div class="section-eyebrow">System Architecture</div>
    <div class="section-title" style="margin-top:.3rem; margin-bottom:0">Data Flow Pipeline</div>
    <div class="pipe-steps">
        <div class="pipe-step"><div class="pipe-step-icon">📦</div><div class="pipe-step-label">Product Info</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="pipe-step-icon">🔢</div><div class="pipe-step-label">ML Features</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="pipe-step-icon">🌳🚀📈</div><div class="pipe-step-label">RF+XGB+HGB</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="pipe-step-icon">📊</div><div class="pipe-step-label">ML Score</div></div>
        <div class="pipe-arrow" style="color:#CBD5E0;font-size:1.8rem">⊕</div>
        <div class="pipe-step"><div class="pipe-step-icon">🖼️</div><div class="pipe-step-label">Product Image</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="pipe-step-icon">🧠</div><div class="pipe-step-label">MobileNetV2</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="pipe-step-icon">🎯</div><div class="pipe-step-label">CNN Score</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step" style="border-color:rgba(232,93,4,0.3);background:rgba(232,93,4,0.05)">
            <div class="pipe-step-icon">🛒</div>
            <div class="pipe-step-label" style="color:#E85D04">Buy Score</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# STATS
st.markdown("""
<div class="stat-grid">
    <div class="stat-box"><div class="stat-val">1.4M</div><div class="stat-label">Amazon Products</div></div>
    <div class="stat-box"><div class="stat-val">3</div><div class="stat-label">Ensemble Models</div></div>
    <div class="stat-box"><div class="stat-val">9</div><div class="stat-label">ML Features</div></div>
    <div class="stat-box"><div class="stat-val">V2</div><div class="stat-label">MobileNet</div></div>
    <div class="stat-box"><div class="stat-val">96²</div><div class="stat-label">Image Size (px)</div></div>
    <div class="stat-box"><div class="stat-val">2</div><div class="stat-label">Training Phases</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
