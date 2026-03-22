import streamlit as st
import os
import json

st.set_page_config(page_title="CNN Model Info", page_icon="🧠", layout="wide")

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
    color: #2D3250 !important;
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
    font-size: 1.7rem; font-weight: 700; color: #9D6FFF;
}
.metric-label {
    font-family: 'Sarabun', sans-serif;
    font-size: .75rem; color: #64748B; margin-top: .2rem;
}

.panel {
    background: #1A1D2E; border: 1px solid #2D3250;
    border-top: 3px solid #9D6FFF;
    border-radius: 16px; padding: 1.8rem; margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.panel.orange-top { border-top-color: #FF6B35; }
.panel.blue-top   { border-top-color: #60A5FA; }
.panel.green-top  { border-top-color: #34D399; }

.panel-title {
    font-family: 'Prompt', sans-serif;
    font-size: .85rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #64748B;
    margin-bottom: 1.4rem; display: flex; align-items: center; gap: .5rem;
}
.panel-title::after { content: ''; flex: 1; height: 1px; background: #2D3250; }

/* Theory section */
.theory-box {
    background: #1E2235; border: 1px solid #2D3250;
    border-radius: 12px; padding: 1.3rem; margin-bottom: 1rem;
    font-family: 'Sarabun', sans-serif;
}
.theory-box-title {
    font-family: 'Prompt', sans-serif;
    font-size: 1rem; font-weight: 600; color: #F0F4FF; margin-bottom: .6rem;
}
.theory-box-text {
    font-size: .88rem; color: #94A3B8; line-height: 1.8;
}
.theory-box-text b { color: #F0F4FF; }
.theory-box-text code {
    background: #252A3C; color: #9D6FFF; padding: .1rem .4rem;
    border-radius: 4px; font-size: .82rem;
}

.arch-table { width: 100%; border-collapse: collapse; }
.arch-table th {
    font-family: 'Sarabun', sans-serif;
    font-size: .72rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; color: #64748B;
    padding: .6rem .8rem; border-bottom: 2px solid #2D3250; text-align: left;
}
.arch-table td {
    font-family: 'Sarabun', sans-serif;
    font-size: .86rem; padding: .75rem .8rem;
    border-bottom: 1px solid #2D3250; color: #94A3B8;
}
.arch-table tr:last-child td { border-bottom: none; }
.arch-table td:first-child { color: #F0F4FF; font-weight: 600; font-family: 'Prompt', sans-serif; }
.arch-table tr.highlight td { background: rgba(157,111,255,0.08); }

.phase-card {
    background: #1E2235; border: 1px solid #2D3250;
    border-radius: 12px; padding: 1.3rem 1.5rem; margin-bottom: .8rem;
}
.phase-num {
    font-family: 'Sarabun', sans-serif;
    font-size: .68rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #9D6FFF; margin-bottom: .4rem;
}
.phase-title {
    font-family: 'Prompt', sans-serif;
    font-size: .95rem; font-weight: 600; color: #F0F4FF; margin-bottom: .6rem;
}
.phase-detail {
    font-family: 'Sarabun', sans-serif;
    font-size: .86rem; color: #94A3B8; line-height: 1.8;
}
.phase-detail b { color: #F0F4FF; }

.aug-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr)); gap: .7rem; }
.aug-card {
    background: #1E2235; border: 1px solid #2D3250;
    border-radius: 10px; padding: .85rem 1rem;
}
.aug-name {
    font-family: 'Prompt', sans-serif;
    font-size: .85rem; font-weight: 600; color: #F0F4FF; margin-bottom: .25rem;
}
.aug-val {
    font-family: 'Sarabun', sans-serif;
    font-size: .8rem; color: #9D6FFF;
}
.aug-explain {
    font-family: 'Sarabun', sans-serif;
    font-size: .76rem; color: #64748B; margin-top: .2rem; line-height: 1.5;
}

.not-trained {
    background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.25);
    border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 1.5rem;
    font-family: 'Sarabun', sans-serif;
}
.not-trained b { color: #FBBF24; }

.class-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap: .5rem; }
.class-pill {
    background: #1E2235; border: 1px solid #2D3250;
    border-radius: 8px; padding: .5rem .8rem;
    font-family: 'Sarabun', sans-serif;
    font-size: .82rem; color: #94A3B8; text-align: center;
}

.flow-text {
    font-family: 'Sarabun', sans-serif;
    font-size: .88rem; color: #94A3B8; line-height: 1.8;
}
.flow-text b { color: #F0F4FF; }

label { color: #64748B !important; font-family: 'Sarabun', sans-serif !important; font-size: .88rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-eyebrow">Deep Learning · Image Classification</div>
    <div class="page-title">CNN Model — Product Image Classifier</div>
    <div class="page-sub">MobileNetV2 จำแนกหมวดหมู่สินค้า Amazon จากรูปภาพ — Transfer Learning 2-Phase<br>เรียนรู้ feature จาก ImageNet แล้ว fine-tune ให้เหมาะกับสินค้า e-commerce</div>
</div>
""", unsafe_allow_html=True)

ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path  = os.path.join(ROOT, 'models', 'product_image_model_best.keras')
class_path  = os.path.join(ROOT, 'models', 'product_image_model_classes.json')
trained     = os.path.exists(model_path)

class_names = []
if os.path.exists(class_path):
    with open(class_path) as f:
        class_names = list(json.load(f).keys())

st.markdown(f"""
<div class="metric-row">
    <div class="metric-box"><div class="metric-val">V2</div><div class="metric-label">MobileNet Version</div></div>
    <div class="metric-box"><div class="metric-val">96²</div><div class="metric-label">Input Size (px)</div></div>
    <div class="metric-box"><div class="metric-val">3.4M</div><div class="metric-label">Base Parameters</div></div>
    <div class="metric-box"><div class="metric-val">30</div><div class="metric-label">Fine-tune Layers</div></div>
    <div class="metric-box"><div class="metric-val">15+20</div><div class="metric-label">Max Epochs</div></div>
    <div class="metric-box"><div class="metric-val">{len(class_names) if class_names else 'N/A'}</div><div class="metric-label">Product Categories</div></div>
</div>
""", unsafe_allow_html=True)

if not trained:
    st.markdown("""
    <div class="not-trained">
        ⚠️ <b>ยังไม่ได้เทรนโมเดล</b> — รัน <code>python train_image_model.py</code> เพื่อเทรน<br>
        ต้องวางรูปภาพใน <code>data/images/&lt;category&gt;/</code> ก่อน
    </div>
    """, unsafe_allow_html=True)

# ── CNN THEORY ────────────────────────────────────────────
st.markdown('<div class="panel"><div class="panel-title">ทฤษฎี Convolutional Neural Network (CNN)</div>', unsafe_allow_html=True)
st.markdown("""
<div class="theory-box">
    <div class="theory-box-title">CNN คืออะไร และทำงานอย่างไร?</div>
    <div class="theory-box-text">
        Convolutional Neural Network (CNN) คือสถาปัตยกรรม Neural Network ที่ออกแบบมาเพื่อประมวลผล
        <strong>ข้อมูลที่มีโครงสร้างตาราง</strong> เช่น รูปภาพ โดยใช้ <strong>Convolution Layer</strong>
        ที่มี filter เล็กๆ เลื่อนผ่านรูปภาพเพื่อตรวจจับ pattern<br><br>
        Layer ต้นๆ จะตรวจจับ feature พื้นฐาน เช่น <strong>เส้นตรง, สี, มุม</strong>
        Layer ลึกขึ้นจะรวม pattern เหล่านั้นเป็น feature ที่ซับซ้อน เช่น
        <strong>รูปทรง, วัตถุ, และ context ของรูป</strong><br><br>
        ข้อดีของ CNN เมื่อเทียบกับ Fully Connected: <strong>parameter sharing</strong>
        (filter เดียวใช้กับทุกตำแหน่ง) และ <strong>translation invariance</strong>
        (จำสิ่งของได้แม้อยู่คนละตำแหน่งในรูป)
    </div>
</div>
<div class="theory-box">
    <div class="theory-box-title">Transfer Learning คืออะไร?</div>
    <div class="theory-box-text">
        Transfer Learning คือการนำ <strong>โมเดลที่เทรนแล้วจาก dataset ขนาดใหญ่</strong>
        มาใช้เป็นจุดเริ่มต้น แทนที่จะเทรนจากศูนย์<br><br>
        ในโปรเจกต์นี้ใช้ <strong>MobileNetV2 ที่เทรนบน ImageNet</strong>
        (รูปภาพ 14 ล้านรูป, 1,000 category) ซึ่งโมเดลได้เรียนรู้วิธีดึง feature
        จากรูปภาพมาแล้วอย่างดี<br><br>
        เราเพียง "ต่อ" classification head ของเราเองที่ปลาย และ fine-tune
        บางส่วนให้เหมาะกับงาน — <strong>ประหยัดเวลาเทรนและข้อมูล</strong>
        เมื่อเทียบกับการเทรนจากศูนย์
    </div>
</div>
<div class="theory-box">
    <div class="theory-box-title">ทำไมถึงเลือก MobileNetV2?</div>
    <div class="theory-box-text">
        MobileNetV2 ใช้ <strong>Depthwise Separable Convolution</strong> ที่แยก
        spatial convolution และ channel mixing ออกจากกัน ทำให้มี parameter
        น้อยกว่า VGG/ResNet มาก (3.4M vs 25M+) แต่ accuracy ใกล้เคียงกัน<br><br>
        เหมาะกับโปรเจกต์นี้เพราะ: <strong>เร็ว, RAM น้อย, รัน CPU ได้</strong>
        ในขณะที่โมเดลใหญ่อย่าง EfficientNetB3 ต้องการทรัพยากรมากกว่า
        และอาจ timeout บน Streamlit Cloud
    </div>
</div>
</div>
""", unsafe_allow_html=True)

# Architecture
st.markdown('<div class="panel blue-top"><div class="panel-title">Network Architecture</div>', unsafe_allow_html=True)
st.markdown("""
<table class="arch-table">
<tr><th>Layer</th><th>Output Shape</th><th>หน้าที่</th></tr>
<tr><td>Input</td><td>96 × 96 × 3</td><td>รับรูปภาพ RGB ขนาด 96×96 px (ลดจาก 224 เพื่อความเร็ว)</td></tr>
<tr><td>MobileNetV2 (base)</td><td>3 × 3 × 1280</td><td>ดึง feature ระดับสูงด้วย ImageNet pretrained weights (3.4M params, frozen Phase 1)</td></tr>
<tr><td>GlobalAveragePooling2D</td><td>1280</td><td>บีบ spatial dimension (3×3) → vector 1280 มิติ ลด parameter</td></tr>
<tr><td>BatchNormalization</td><td>1280</td><td>Normalize activations ให้การเทรนเสถียร ลด internal covariate shift</td></tr>
<tr class="highlight"><td>Dense(256) + ReLU + L2</td><td>256</td><td>เรียนรู้ feature เฉพาะสินค้า e-commerce + L2(1e-4) ป้องกัน overfitting</td></tr>
<tr class="highlight"><td>Dropout(0.4)</td><td>256</td><td>ปิด 40% neurons แบบสุ่มระหว่าง training — regularization หลัก</td></tr>
<tr class="highlight"><td>Dense(128) + ReLU + L2</td><td>128</td><td>ลด dimensionality ต่อเนื่อง + L2(1e-4) regularization</td></tr>
<tr class="highlight"><td>Dropout(0.3)</td><td>128</td><td>ปิด 30% neurons แบบสุ่มก่อน output layer</td></tr>
<tr><td>Dense(N) + Softmax</td><td>N classes</td><td>ความน่าจะเป็นสำหรับแต่ละ category (N = จำนวน category ที่เทรน)</td></tr>
</table>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Training phases
st.markdown('<div class="panel"><div class="panel-title">กลยุทธ์การเทรน — 2 Phases</div>', unsafe_allow_html=True)
st.markdown("""
<div style="background:#1E2235; border:1px solid #2D3250; border-radius:12px; padding:1rem 1.4rem; margin-bottom:1rem; font-family:'Sarabun',sans-serif; font-size:.88rem; color:#94A3B8; line-height:1.8;">
    <strong style="font-family:'Prompt',sans-serif; color:#E2E8F0;">ทำไมต้องใช้ 2-Phase Training?</strong><br>
    ถ้า fine-tune ทั้งโมเดลตั้งแต่ต้นด้วย learning rate สูง จะทำลาย pretrained weights ที่ดีอยู่แล้ว
    (catastrophic forgetting) การแยกเป็น 2 phase ช่วยให้ custom head เรียนรู้ก่อน
    แล้วค่อย fine-tune base ด้วย learning rate ต่ำมาก
</div>
<div class="phase-card">
    <div class="phase-num">Phase 1 — Head Training</div>
    <div class="phase-title">Base Frozen (ล็อก MobileNetV2 ทั้งหมด)</div>
    <div class="phase-detail">
        <b>Learning Rate:</b> 1e-3 &nbsp;|&nbsp; <b>Epochs:</b> สูงสุด 20 + EarlyStopping(patience=8)<br>
        <b>Optimizer:</b> Adam &nbsp;|&nbsp; <b>Loss:</b> CategoricalCrossentropy(label_smoothing=0.05)<br><br>
        ล็อก MobileNetV2 ทั้งหมด เทรนเฉพาะ Dense layers ที่เราเพิ่มเข้าไป
        ให้โมเดลเรียนรู้การจำแนก category สินค้าโดยใช้ feature จาก ImageNet
        เป็น "warm start" ก่อน fine-tune ใน Phase 2
    </div>
</div>
<div class="phase-card">
    <div class="phase-num">Phase 2 — Fine-tuning</div>
    <div class="phase-title">Top 30 Layers Unfrozen (fine-tune บางส่วน)</div>
    <div class="phase-detail">
        <b>Learning Rate:</b> 1e-5 &nbsp;|&nbsp; <b>Epochs:</b> สูงสุด 20 + EarlyStopping(patience=5)<br>
        <b>Optimizer:</b> Adam(1e-5) — เล็กมากเพื่อไม่ทำลาย pretrained weights<br><br>
        Unfreeze 30 layer ท้ายสุดของ MobileNetV2 ให้ fine-tune พร้อม head
        ล็อก layer ต้น (feature extractor ระดับพื้นฐาน: edge, texture) ไว้ตามเดิม
        ใช้ ReduceLROnPlateau ลด lr อัตโนมัติเมื่อ val_loss หยุดพัฒนา
    </div>
</div>
</div>
""", unsafe_allow_html=True)

# Data Augmentation
st.markdown('<div class="panel orange-top"><div class="panel-title">Data Augmentation (Training Only)</div>', unsafe_allow_html=True)
st.markdown("""
<div style="background:rgba(255,107,53,0.06); border:1px solid rgba(255,107,53,0.2); border-radius:10px; padding:.9rem 1.2rem; margin-bottom:1rem; font-family:'Sarabun',sans-serif; font-size:.88rem; color:#94A3B8; line-height:1.7;">
    <strong style="font-family:'Prompt',sans-serif; color:#E2E8F0;">Data Augmentation คืออะไร?</strong><br>
    การสร้างรูปภาพ "แปลงร่าง" จากรูปต้นฉบับ เช่น หมุน, พลิก, ซูม เพื่อให้โมเดลเห็นรูปแบบที่หลากหลาย
    ช่วยแก้ปัญหา overfitting เมื่อ dataset มีขนาดเล็ก และทำให้โมเดล robust ต่อรูปภาพในชีวิตจริง
    ทำเฉพาะตอน training เท่านั้น — validation/test ใช้รูปต้นฉบับ
</div>
""", unsafe_allow_html=True)

augs = [
    ("RandomFlip", "Horizontal + Vertical", "พลิกรูปซ้าย-ขวา และ บน-ล่าง"),
    ("RandomRotation", "±20°", "หมุนรูปสุ่มในช่วง ±20 องศา"),
    ("RandomZoom", "±15%", "ซูมเข้า-ออกสุ่มในช่วง ±15%"),
    ("RandomContrast", "±20%", "ปรับ contrast สุ่มในช่วง ±20%"),
    ("RandomBrightness", "±15%", "ปรับความสว่างสุ่มในช่วง ±15%"),
    ("Rescaling", "÷255 (normalize)", "แปลง pixel 0–255 → 0–1 ก่อนป้อน MobileNetV2"),
]
st.markdown('<div class="aug-grid">', unsafe_allow_html=True)
for name, val, explain in augs:
    st.markdown(f"""
    <div class="aug-card">
        <div class="aug-name">{name}</div>
        <div class="aug-val">{val}</div>
        <div class="aug-explain">{explain}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
<div style="margin-top:1rem; font-family:'Sarabun',sans-serif; font-size:.85rem; color:#4A5568; line-height:1.7;">
    Augmentation ทำบน <strong>GPU pipeline (tf.data)</strong> ทำให้ไม่เพิ่มเวลา I/O<br>
    <strong>label_smoothing=0.05</strong> ช่วยลด overconfidence โดยเฉพาะ category ที่รูปคล้ายกัน
    แทนที่จะ target probability = 1.0 จะใช้ 0.95 แทน ทำให้ gradient ไม่ saturate
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Category list (if trained)
if class_names:
    st.markdown('<div class="panel green-top"><div class="panel-title">Product Categories ที่เทรนแล้ว</div>', unsafe_allow_html=True)
    st.markdown('<div class="class-grid">', unsafe_allow_html=True)
    for c in sorted(class_names):
        st.markdown(f'<div class="class-pill">📦 {c}</div>', unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

# How CNN score works
st.markdown("""
<div class="panel">
<div class="panel-title">CNN Score ทำงานอย่างไร?</div>
<div class="flow-text">
เมื่อผู้ใช้อัปโหลดรูปสินค้า CNN จะ output probability สำหรับทุก category<br>
<br>
<b>CNN Score = ความมั่นใจสูงสุด (top-1 confidence)</b><br>
• ถ้า CNN มั่นใจสูง (≥70%) → รูปภาพชัดเจน คุณภาพดี → เป็น positive signal<br>
• ถ้า CNN มั่นใจปานกลาง (40–69%) → รูปภาพโอเคแต่อาจมีความกำกวม<br>
• ถ้า CNN มั่นใจต่ำ (&lt;40%) → รูปภาพไม่ชัด ไม่ตรง category → เป็น warning signal<br>
<br>
<b>Final Buy Score = ML_score × 70% + CNN_score × 30%</b><br>
• ML score มีน้ำหนักมากกว่า เพราะใช้ข้อมูลจริง (ราคา, รีวิว, ดาว)<br>
• CNN score เป็น bonus indicator ด้านคุณภาพรูปภาพสินค้า
</div>
</div>
""", unsafe_allow_html=True)

st.code("python train_image_model.py", language="bash")

# References
st.markdown("""
<div class="panel">
<div class="panel-title">แหล่งอ้างอิง (References)</div>
<div class="flow-text" style="line-height:2.2">
<b style="color:#E2E8F0; font-family:'Prompt',sans-serif;">Dataset</b><br>
• Rashad, A. E. (2023). <i>Amazon Products Image</i>. Kaggle.
  <span style="color:#A0AEC0">— https://www.kaggle.com/datasets/ahmedelsayedrashad/amazon-products-image</span><br>
<br>
<b style="color:#E2E8F0; font-family:'Prompt',sans-serif;">Neural Network Architecture</b><br>
• Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., &amp; Chen, L.-C. (2018).
  MobileNetV2: Inverted Residuals and Linear Bottlenecks. <i>CVPR 2018</i>.
  doi:10.1109/CVPR.2018.00474<br>
• Tan, M., &amp; Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs.
  <i>ICML 2019</i>. arXiv:1905.11946<br>
• LeCun, Y., Bengio, Y., &amp; Hinton, G. (2015). Deep Learning. <i>Nature</i>, 521, 436–444.<br>
<br>
<b style="color:#E2E8F0; font-family:'Prompt',sans-serif;">Transfer Learning &amp; Training Techniques</b><br>
• Yosinski, J., et al. (2014). How Transferable are Features in Deep Neural Networks?
  <i>NeurIPS 2014</i>. arXiv:1411.1792<br>
• Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
  <i>JMLR</i>, 15(1), 1929–1958.<br>
• Ioffe, S., &amp; Szegedy, C. (2015). Batch Normalization. <i>ICML 2015</i>. arXiv:1502.03167<br>
<br>
<b style="color:#E2E8F0; font-family:'Prompt',sans-serif;">Framework</b><br>
• Abadi, M., et al. (2016). TensorFlow: A System for Large-Scale Machine Learning.
  <i>OSDI 2016</i>. — https://www.tensorflow.org<br>
• Keras Documentation. — https://keras.io
</div>
</div>
""", unsafe_allow_html=True)
