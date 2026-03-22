import streamlit as st
import numpy as np
import os
import json
from PIL import Image

st.set_page_config(page_title="ทดสอบ CNN Model", page_icon="🧠", layout="wide")

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
[data-testid="stDecoration"], header { display: none !important; }

.page-header {
    padding: 1.5rem 0 2rem;
    border-bottom: 1px solid #2D3250;
    margin-bottom: 2rem;
}
.page-eyebrow {
    font-family: 'Sarabun', sans-serif;
    font-size: .7rem; font-weight: 700; letter-spacing: .15em;
    text-transform: uppercase; color: #7C3AED; margin-bottom: .5rem;
}
.page-title {
    font-family: 'Prompt', sans-serif;
    font-size: 2rem; font-weight: 700; color: #F0F4FF;
}
.page-sub {
    font-family: 'Sarabun', sans-serif;
    font-size: .95rem; color: #94A3B8; margin-top: .4rem; line-height: 1.6;
}

/* Instruction panel */
.instruction-panel {
    background: #1A1D2E; border: 1px solid #2D3250;
    border-top: 3px solid #9D6FFF;
    border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.instruction-title {
    font-family: 'Prompt', sans-serif;
    font-size: .95rem; font-weight: 600; color: #F0F4FF; margin-bottom: .8rem;
}
.instruction-steps { list-style: none; padding: 0; margin: 0; }
.instruction-steps li {
    font-family: 'Sarabun', sans-serif;
    font-size: .88rem; color: #94A3B8; line-height: 1.6;
    padding: .3rem 0; display: flex; gap: .7rem; align-items: flex-start;
}
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%;
    background: rgba(157,111,255,0.15); color: #9D6FFF;
    font-size: .72rem; font-weight: 700; flex-shrink: 0; margin-top: .1rem;
    font-family: 'Prompt', sans-serif;
}

.panel {
    background: #1A1D2E; border: 1px solid #2D3250;
    border-top: 3px solid #9D6FFF;
    border-radius: 16px; padding: 1.8rem; margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.panel.orange-top { border-top-color: #E85D04; }
.panel.neutral-top { border-top-color: #475569; }

.panel-title {
    font-family: 'Prompt', sans-serif;
    font-size: .82rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #64748B;
    margin-bottom: 1.2rem; display: flex; align-items: center; gap: .5rem;
}
.panel-title::after { content: ''; flex: 1; height: 1px; background: #2D3250; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #9D6FFF, #7C3AED) !important;
    border: none !important; border-radius: 12px !important; color: white !important;
    font-family: 'Sarabun', sans-serif !important; font-weight: 700 !important;
    font-size: .95rem !important; padding: .85rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.25) !important;
    transition: all .2s !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(124,58,237,0.35) !important;
    transform: translateY(-1px) !important;
}

/* Top prediction card */
.top-wrap {
    background: linear-gradient(135deg, #1A152E, #14102A);
    border: 1px solid rgba(157,111,255,0.3);
    border-radius: 16px; padding: 2rem 1.8rem; text-align: center;
    position: relative; overflow: hidden; margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(124,58,237,0.10);
}
.top-wrap::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 50% at 50% 0%, rgba(157,111,255,0.15) 0%, transparent 70%);
}
.top-label {
    font-family: 'Sarabun', sans-serif;
    font-size: .72rem; font-weight: 700; letter-spacing: .15em;
    text-transform: uppercase; color: rgba(157,111,255,0.8); margin-bottom: .5rem;
}
.top-value {
    font-family: 'Prompt', sans-serif;
    font-size: 2rem; font-weight: 700; line-height: 1.2;
    margin-bottom: .3rem; color: #F0F4FF;
}
.top-conf {
    font-family: 'Sarabun', sans-serif;
    font-size: 1.1rem; font-weight: 700;
}
.top-sub {
    font-family: 'Sarabun', sans-serif;
    font-size: .82rem; color: #475569; margin-top: .3rem;
}

/* Confidence meaning */
.conf-explain-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: .7rem; margin: .8rem 0; }
.conf-explain-card {
    background: #1E2235; border: 1px solid #2D3250;
    border-radius: 10px; padding: .8rem; text-align: center;
    font-family: 'Sarabun', sans-serif;
}
.conf-badge {
    display: inline-block; padding: .2rem .7rem; border-radius: 999px;
    font-size: .78rem; font-weight: 600; margin-bottom: .3rem;
}
.conf-high { background: rgba(52,211,153,0.14); color: #34D399; }
.conf-med  { background: rgba(251,191,36,0.14); color: #FBBF24; }
.conf-low  { background: rgba(248,113,113,0.14); color: #F87171; }
.conf-range { font-size: .78rem; font-weight: 600; color: #2D3748; margin-bottom: .2rem; }
.conf-desc  { font-size: .74rem; color: #64748B; line-height: 1.5; }

/* probability bars */
.prob-row { margin-bottom: .8rem; }
.prob-header {
    display: flex; justify-content: space-between;
    font-family: 'Sarabun', sans-serif;
    font-size: .86rem; margin-bottom: .3rem;
}
.prob-label { font-weight: 600; color: #2D3748; }
.prob-val   { color: #9D6FFF; font-weight: 700; }
.prob-bar-bg { background: #252A3C; border-radius: 999px; height: 8px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 999px; }

.placeholder {
    background: #1A1D2E; border: 2px dashed #3D4370;
    border-radius: 16px; padding: 4rem 2rem; text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.placeholder-icon { font-size: 2.5rem; opacity: .4; margin-bottom: 1rem; }
.placeholder-text {
    font-family: 'Sarabun', sans-serif;
    font-size: .95rem; color: #475569; line-height: 1.7;
}

.info-pill {
    display: inline-flex; gap: .4rem; align-items: center;
    background: rgba(157,111,255,0.10); border: 1px solid rgba(157,111,255,0.2);
    border-radius: 8px; padding: .4rem .8rem;
    font-family: 'Sarabun', sans-serif;
    font-size: .78rem; color: #9D6FFF; margin: .2rem;
}

label {
    color: #64748B !important;
    font-family: 'Sarabun', sans-serif !important;
    font-size: .88rem !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-eyebrow">ทดสอบโมเดล · Neural Network</div>
    <div class="page-title">ทดสอบ CNN Model</div>
    <div class="page-sub">อัปโหลดรูปสินค้า → MobileNetV2 จำแนกหมวดหมู่จากรูปภาพ (Image Classification)<br>โมเดลแสดง Top-5 predictions พร้อม confidence score สำหรับแต่ละ category</div>
</div>
""", unsafe_allow_html=True)

ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, 'models')

@st.cache_resource
def load_cnn():
    try:
        import tensorflow as tf
        # จำกัด thread เพื่อไม่ให้ TF ยึด CPU ทั้งหมด → Streamlit ไม่ disconnect
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        mp = os.path.join(MODELS_DIR, 'product_image_model_best.keras')
        cp = os.path.join(MODELS_DIR, 'product_image_model_classes.json')
        if not os.path.exists(mp):
            return None, None
        model     = tf.keras.models.load_model(mp)
        class_map = None
        if os.path.exists(cp):
            with open(cp) as f:
                d = json.load(f)
            class_map = {v: k for k, v in d.items()}
        return model, class_map
    except Exception as e:
        st.warning(f"โหลด CNN ไม่ได้: {e}")
        return None, None

cnn_model, class_map = load_cnn()

def cnn_predict(img):
    if cnn_model is None:
        return []
    try:
        img_size = cnn_model.input_shape[1]
        img_arr  = np.expand_dims(
            np.array(img.convert('RGB').resize((img_size, img_size)), dtype=np.float32), 0
        )
        probs = cnn_model.predict(img_arr, verbose=0)[0]
        idxs  = np.argsort(probs)[::-1][:5]
        return [(class_map.get(i, f"class_{i}") if class_map else f"class_{i}", float(probs[i])) for i in idxs]
    except Exception as e:
        st.error(f"CNN error: {e}")
        return []

# Instruction panel
st.markdown("""
<div class="instruction-panel">
    <div class="instruction-title">วิธีใช้งาน</div>
    <ul class="instruction-steps">
        <li><span class="step-num">1</span>คลิก "Browse files" หรือลากรูปภาพสินค้ามาวางในกล่องอัปโหลด (รองรับ JPG, PNG, WebP)</li>
        <li><span class="step-num">2</span>รูปจะแสดงตัวอย่างในช่องด้านซ้าย</li>
        <li><span class="step-num">3</span>กดปุ่ม "จำแนกรูปภาพ" เพื่อให้ CNN วิเคราะห์</li>
        <li><span class="step-num">4</span>ดูผลลัพธ์ Top-1 prediction พร้อม confidence score และ Top-5 ทั้งหมด</li>
        <li><span class="step-num">5</span>ยิ่ง confidence สูง = โมเดลมั่นใจมากในการจำแนก category นั้น</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Layout
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown('<div class="panel"><div class="panel-title">อัปโหลดรูปสินค้า</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("เลือกรูปภาพสินค้า", type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed")
    img = None
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True, caption="รูปที่อัปโหลด")

    if cnn_model is None:
        st.warning("⚠️ ไม่พบ CNN model — รัน `python train_image_model.py` ก่อน")
    else:
        img_size = cnn_model.input_shape[1]
        st.markdown(f"""
        <div style="margin-top:1rem">
            <span class="info-pill">📐 Input: {img_size}×{img_size} px</span>
            <span class="info-pill">🧠 MobileNetV2</span>
            <span class="info-pill">🗂️ {len(class_map) if class_map else '?'} categories</span>
        </div>
        <div style="margin-top:.8rem; font-family:'Sarabun',sans-serif; font-size:.82rem; color:#718096; line-height:1.6;">
            รูปภาพจะถูก resize เป็น {img_size}×{img_size} px อัตโนมัติก่อนป้อนเข้าโมเดล<br>
            แนะนำใช้รูปที่ชัดเจน ไม่มีพื้นหลังรกรุงรัง และเห็นสินค้าชัดเจน
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    btn = st.button("🧠 จำแนกรูปภาพ", type="primary", use_container_width=True,
                    disabled=(cnn_model is None or uploaded is None))

with col_right:
    st.markdown('<div class="panel"><div class="panel-title">ผลการจำแนกหมวดหมู่</div>', unsafe_allow_html=True)

    if btn and uploaded and cnn_model and img:
        with st.spinner("กำลังวิเคราะห์รูปภาพ..."):
            results = cnn_predict(img)

        if results:
            top_cat, top_prob = results[0]
            conf_pct = top_prob * 100
            conf_color = '#059669' if conf_pct >= 70 else ('#B45309' if conf_pct >= 40 else '#DC2626')

            st.markdown(f"""
            <div class="top-wrap">
                <div class="top-label">Top Prediction</div>
                <div class="top-value">{top_cat}</div>
                <div class="top-conf" style="color:{conf_color}">{conf_pct:.1f}% confidence</div>
                <div class="top-sub">จาก {len(results)} หมวดหมู่ที่มีโอกาสสูงสุด</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence explanation
            st.markdown("""
            <div style="margin-bottom:.8rem; font-family:'Prompt',sans-serif; font-size:.82rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#718096;">ความหมาย Confidence Score</div>
            <div class="conf-explain-grid">
                <div class="conf-explain-card">
                    <div class="conf-badge conf-high">สูง ≥70%</div>
                    <div class="conf-range">มั่นใจมาก</div>
                    <div class="conf-desc">โมเดลจำแนก category ได้ชัดเจน รูปภาพคุณภาพดี</div>
                </div>
                <div class="conf-explain-card">
                    <div class="conf-badge conf-med">กลาง 40–69%</div>
                    <div class="conf-range">มั่นใจปานกลาง</div>
                    <div class="conf-desc">โมเดลพอจำแนกได้ แต่มีความกำกวมระหว่าง category</div>
                </div>
                <div class="conf-explain-card">
                    <div class="conf-badge conf-low">ต่ำ &lt;40%</div>
                    <div class="conf-range">ไม่แน่ใจ</div>
                    <div class="conf-desc">รูปไม่ชัด หรือ category ไม่อยู่ในชุดเทรน</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="panel-title" style="margin-top:.5rem">Top 5 Predictions</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'Sarabun',sans-serif; font-size:.82rem; color:#718096; margin-bottom:.8rem; line-height:1.6;">
                Top-5 คือ 5 หมวดหมู่ที่โมเดลคิดว่า "น่าจะใช่" มากที่สุด
                เรียงจากความมั่นใจสูงไปต่ำ ผลรวมทุก category = 100%
            </div>
            """, unsafe_allow_html=True)
            for rank, (cat, prob) in enumerate(results, 1):
                pct = prob * 100
                if rank == 1:
                    bar_color = 'linear-gradient(90deg,#9D6FFF,#7C3AED)'
                    label_color = '#E2E8F0'
                    val_color = '#9D6FFF'
                else:
                    bar_color = 'linear-gradient(90deg,#374151,#2D3250)'
                    label_color = '#64748B'
                    val_color = '#475569'
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-header">
                        <span class="prob-label" style="color:{label_color}">#{rank} {cat}</span>
                        <span class="prob-val" style="color:{val_color}">{pct:.1f}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{bar_color}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("ไม่สามารถวิเคราะห์รูปภาพได้")
    else:
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">🧠</div>
            <div class="placeholder-text">
                อัปโหลดรูปสินค้าทางซ้าย<br>
                แล้วกด <b style="color:#7C3AED">จำแนกรูปภาพ</b><br><br>
                <span style="font-size:.85rem; color:#475569">
                    CNN จะทำนายว่าสินค้าในรูป<br>
                    อยู่ใน category ไหน พร้อม Top-5
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Bottom explanation panel
st.markdown("""
<div class="panel neutral-top" style="margin-top:1.5rem">
    <div class="panel-title">Top-5 Predictions คืออะไร และ Confidence Score หมายความว่าอย่างไร?</div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.2rem;">
        <div style="background:#F7FAFC; border:1px solid #2D3250; border-radius:12px; padding:1.1rem;">
            <div style="font-family:'Prompt',sans-serif; font-size:.9rem; font-weight:600; color:#1A202C; margin-bottom:.5rem;">Top-5 Predictions</div>
            <div style="font-family:'Sarabun',sans-serif; font-size:.86rem; color:#4A5568; line-height:1.7;">
                CNN ไม่ได้บอกแค่ "คำตอบเดียว" แต่ให้ probability สำหรับทุก category
                Top-5 คือ 5 category ที่ได้ probability สูงสุด<br><br>
                มีประโยชน์เมื่อ: รูปมีความกำกวม เช่น กล้องอาจถูกจำแนกเป็น
                "Electronics" หรือ "Camera" ขึ้นอยู่กับการเทรน
                Top-5 ช่วยให้เห็นว่าโมเดล "กำลังลังเล" อยู่กับ category ใด
            </div>
        </div>
        <div style="background:#F7FAFC; border:1px solid #2D3250; border-radius:12px; padding:1.1rem;">
            <div style="font-family:'Prompt',sans-serif; font-size:.9rem; font-weight:600; color:#1A202C; margin-bottom:.5rem;">Confidence Score</div>
            <div style="font-family:'Sarabun',sans-serif; font-size:.86rem; color:#4A5568; line-height:1.7;">
                Confidence Score คือ probability (0–100%) ที่ Softmax layer ให้กับ category นั้น
                ยิ่งสูง = โมเดลมั่นใจมากกว่า<br><br>
                <strong>ข้อควรระวัง:</strong> confidence สูงไม่ได้แปลว่าถูกเสมอ
                โมเดล CNN สามารถ "overconfident" ได้ โดยเฉพาะถ้ารูปแตกต่างจาก training data
                จึงใช้ label_smoothing=0.05 ลด overconfidence ในการเทรน
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
