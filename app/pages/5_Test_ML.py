import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib

st.set_page_config(page_title="ทดสอบ ML — ทำนายยอดขาย", page_icon="📦", layout="wide", initial_sidebar_state="expanded")

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

/* Instruction panel */
.instruction-panel {
    background: #1A1D2E; border: 1px solid #2D3250;
    border-top: 3px solid #FF6B35;
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
    background: rgba(255,107,53,0.15); color: #FF6B35;
    font-size: .72rem; font-weight: 700; flex-shrink: 0; margin-top: .1rem;
    font-family: 'Prompt', sans-serif;
}

.panel {
    background: #1A1D2E; border: 1px solid #2D3250;
    border-top: 3px solid #FF6B35;
    border-radius: 16px; padding: 1.8rem; margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.panel.purple-top { border-top-color: #9D6FFF; }

.panel-title {
    font-family: 'Prompt', sans-serif;
    font-size: .82rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #64748B;
    margin-bottom: 1.2rem; display: flex; align-items: center; gap: .5rem;
}
.panel-title::after { content: ''; flex: 1; height: 1px; background: #2D3250; }

/* Field explanation */
.field-explain {
    background: #1E2235; border: 1px solid #2D3250;
    border-radius: 10px; padding: .7rem 1rem; margin: .3rem 0 .8rem;
    font-family: 'Sarabun', sans-serif; font-size: .82rem; color: #64748B; line-height: 1.6;
}
.field-explain b { color: #94A3B8; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #FF6B35, #E85D04) !important;
    border: none !important; border-radius: 12px !important; color: white !important;
    font-family: 'Sarabun', sans-serif !important; font-weight: 700 !important;
    font-size: .95rem !important; padding: .85rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(232,93,4,0.25) !important;
    transition: all .2s !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(232,93,4,0.35) !important;
    transform: translateY(-1px) !important;
}

/* result card */
.result-wrap {
    border-radius: 16px; padding: 2rem 1.8rem; text-align: center;
    position: relative; overflow: hidden; margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.result-label {
    font-family: 'Sarabun', sans-serif;
    font-size: .72rem; font-weight: 700; letter-spacing: .15em;
    text-transform: uppercase; margin-bottom: .6rem;
}
.result-tier {
    font-family: 'Prompt', sans-serif;
    font-size: 3rem; font-weight: 700; line-height: 1; margin-bottom: .4rem;
}
.result-sub {
    font-family: 'Sarabun', sans-serif;
    font-size: .88rem; line-height: 1.6;
}

.tier-high {
    background: linear-gradient(135deg, #0D2420, #0A1E1B);
    border: 1px solid rgba(52,211,153,0.3);
}
.tier-med {
    background: linear-gradient(135deg, #1E1608, #1A1308);
    border: 1px solid rgba(251,191,36,0.3);
}
.tier-low {
    background: linear-gradient(135deg, #200A0A, #1A0808);
    border: 1px solid rgba(248,113,113,0.3);
}
.tier-label-high { color: rgba(52,211,153,0.8); }
.tier-label-med  { color: rgba(251,191,36,0.8); }
.tier-label-low  { color: rgba(248,113,113,0.8); }

/* Tier explanation boxes */
.tier-explain-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: .8rem; margin-top: .8rem; }
.tier-explain-card {
    background: #1E2235; border: 1px solid #2D3250;
    border-radius: 10px; padding: .9rem; text-align: center;
    font-family: 'Sarabun', sans-serif;
}
.tier-explain-badge {
    display: inline-block; padding: .22rem .7rem; border-radius: 999px;
    font-family: 'Prompt', sans-serif; font-size: .82rem; font-weight: 600;
    margin-bottom: .4rem;
}
.teb-high { background: rgba(52,211,153,0.14); color: #34D399; }
.teb-med  { background: rgba(251,191,36,0.14); color: #FBBF24; }
.teb-low  { background: rgba(248,113,113,0.14); color: #F87171; }
.tier-explain-range {
    font-size: .78rem; font-weight: 600; color: #2D3748; margin-bottom: .3rem;
}
.tier-explain-desc { font-size: .76rem; color: #64748B; line-height: 1.55; }

/* probability bars */
.prob-row { margin-bottom: .8rem; }
.prob-header {
    display: flex; justify-content: space-between;
    font-family: 'Sarabun', sans-serif;
    font-size: .86rem; color: #64748B; margin-bottom: .3rem;
}
.prob-label { font-weight: 700; color: #2D3748; }
.prob-val   { font-weight: 700; }
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

label {
    color: #64748B !important;
    font-family: 'Sarabun', sans-serif !important;
    font-size: .88rem !important;
}

[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] input {
    background: #1E2235 !important; border: 1px solid #2D3250 !important;
    border-radius: 10px !important; color: #E2E8F0 !important;
    font-family: 'Sarabun', sans-serif !important;
}
[data-testid="stNumberInput"] input:focus, [data-testid="stTextInput"] input:focus {
    border-color: #FF6B35 !important;
    box-shadow: 0 0 0 3px rgba(255,107,53,0.15) !important;
}
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
    <div class="page-eyebrow">ทดสอบโมเดล · Machine Learning Ensemble</div>
    <div class="page-title">ทำนายยอดขายสินค้า</div>
    <div class="page-sub">ใส่ข้อมูลสินค้า → Ensemble (RF + XGBoost + HistGBM) ทำนายระดับยอดขายต่อเดือน (Low / Medium / High)<br>โมเดลใช้ราคา, คะแนนดาว, จำนวนรีวิว และหมวดหมู่สินค้าในการตัดสินใจ</div>
</div>
""", unsafe_allow_html=True)

ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, 'models')
USD_TO_THB = 35.0

# ── แปลชื่อหมวดหมู่เป็นภาษาไทย ──────────────────────────────
CATEGORY_TH = {
    # Arts & Crafts
    "Beading & Jewelry Making": "การทำเครื่องประดับและลูกปัด",
    "Fabric Decorating": "การตกแต่งผ้า",
    "Knitting & Crochet Supplies": "ไหมถักและโครเชต์",
    "Printmaking Supplies": "อุปกรณ์การพิมพ์",
    "Scrapbooking & Stamping Supplies": "สครปบุ๊กและการปั๊มตรา",
    "Sewing Products": "อุปกรณ์เย็บผ้า",
    "Craft & Hobby Fabric": "ผ้าสำหรับงานฝีมือ",
    "Needlework Supplies": "อุปกรณ์งานเย็บปักถักร้อย",
    "Arts, Crafts & Sewing Storage": "การจัดเก็บงานฝีมือและเย็บผ้า",
    "Painting, Drawing & Art Supplies": "อุปกรณ์วาดภาพและศิลปะ",
    "Craft Supplies & Materials": "วัสดุงานฝีมือ",
    "Gift Wrapping Supplies": "อุปกรณ์ห่อของขวัญ",
    "Party Decorations": "ของตกแต่งปาร์ตี้",
    "Arts & Crafts Supplies": "อุปกรณ์งานฝีมือ",
    # Automotive
    "Automotive Paint & Paint Supplies": "สีและอุปกรณ์ทาสีรถยนต์",
    "Heavy Duty & Commercial Vehicle Equipment": "อุปกรณ์รถบรรทุกและพาณิชยกรรม",
    "Automotive Tires & Wheels": "ยางและล้อรถยนต์",
    "Automotive Tools & Equipment": "เครื่องมือยานยนต์",
    "Automotive Performance Parts & Accessories": "ชิ้นส่วนสมรรถนะรถยนต์",
    "Car Care": "ผลิตภัณฑ์ดูแลรักษารถยนต์",
    "Oils & Fluids": "น้ำมันและของเหลวรถยนต์",
    "Automotive Replacement Parts": "อะไหล่รถยนต์",
    "Lights, Bulbs & Indicators": "ไฟและสัญญาณรถยนต์",
    "Automotive Interior Accessories": "อุปกรณ์ตกแต่งภายในรถ",
    "Automotive Exterior Accessories": "อุปกรณ์ตกแต่งภายนอกรถ",
    "Automotive Enthusiast Merchandise": "สินค้าสำหรับคนรักรถ",
    "Car Electronics & Accessories": "อิเล็กทรอนิกส์และอุปกรณ์รถยนต์",
    "RV Parts & Accessories": "อะไหล่และอุปกรณ์รถแคมปิ้ง",
    "Motorcycle & Powersports": "มอเตอร์ไซค์และยานพาหนะกีฬา",
    # Baby
    "Baby Diapering Products": "ผ้าอ้อมและอุปกรณ์ทารก",
    "Baby & Toddler Feeding Supplies": "อุปกรณ์ให้อาหารทารกและเด็กเล็ก",
    "Pregnancy & Maternity Products": "สินค้าคุณแม่ตั้งครรภ์และหลังคลอด",
    "Child Safety Car Seats & Accessories": "คาร์ซีทเด็กและอุปกรณ์",
    "Baby Safety Products": "อุปกรณ์ความปลอดภัยสำหรับทารก",
    "Toilet Training Products": "อุปกรณ์ฝึกการใช้ห้องน้ำ",
    "Baby Care Products": "ผลิตภัณฑ์ดูแลทารก",
    "Baby Travel Gear": "อุปกรณ์เดินทางสำหรับทารก",
    "Baby Activity & Entertainment Products": "ของเล่นและกิจกรรมสำหรับทารก",
    "Baby Gifts": "ของขวัญทารก",
    "Baby Stationery": "เครื่องเขียนทารก",
    "Nursery Furniture, Bedding & Décor": "เฟอร์นิเจอร์และของตกแต่งห้องเด็ก",
    "Baby": "สินค้าทารก",
    "Baby Boys' Clothing & Shoes": "เสื้อผ้าและรองเท้าทารกชาย",
    "Baby Strollers & Accessories": "รถเข็นทารกและอุปกรณ์",
    "Baby Girls' Clothing & Shoes": "เสื้อผ้าและรองเท้าทารกหญิง",
    "Baby & Child Care Products": "ผลิตภัณฑ์ดูแลทารกและเด็ก",
    "Baby & Toddler Toys": "ของเล่นทารกและเด็กเล็ก",
    # Beauty
    "Beauty & Personal Care": "ความงามและการดูแลส่วนตัว",
    "Perfumes & Fragrances": "น้ำหอมและกลิ่นหอม",
    "Hair Care Products": "ผลิตภัณฑ์ดูแลผม",
    "Makeup": "เครื่องสำอาง",
    "Skin Care Products": "ผลิตภัณฑ์ดูแลผิว",
    "Beauty Tools & Accessories": "เครื่องมือและอุปกรณ์เสริมความงาม",
    "Foot, Hand & Nail Care Products": "ผลิตภัณฑ์ดูแลมือ เท้า และเล็บ",
    "Personal Care Products": "ผลิตภัณฑ์ดูแลร่างกาย",
    "Shaving & Hair Removal Products": "อุปกรณ์โกนหนวดและกำจัดขน",
    # Computers & Electronics
    "Computer Servers": "เซิร์ฟเวอร์คอมพิวเตอร์",
    "Data Storage": "อุปกรณ์จัดเก็บข้อมูล",
    "Computer Monitors": "จอภาพคอมพิวเตอร์",
    "Computers & Tablets": "คอมพิวเตอร์และแท็บเล็ต",
    "Tablet Replacement Parts": "อะไหล่แท็บเล็ต",
    "Computer Networking": "อุปกรณ์เครือข่ายคอมพิวเตอร์",
    "Computer Components": "ชิ้นส่วนคอมพิวเตอร์",
    "Tablet Accessories": "อุปกรณ์เสริมแท็บเล็ต",
    "Laptop Accessories": "อุปกรณ์เสริมแล็ปท็อป",
    "Computer External Components": "อุปกรณ์ต่อพ่วงคอมพิวเตอร์",
    "Wearable Technology": "เทคโนโลยีสวมใส่",
    "Televisions & Video Products": "โทรทัศน์และผลิตภัณฑ์วิดีโอ",
    "GPS & Navigation": "GPS และระบบนำทาง",
    "Headphones & Earbuds": "หูฟัง",
    "Office Electronics": "อิเล็กทรอนิกส์สำนักงาน",
    "Portable Audio & Video": "เสียงและวิดีโอแบบพกพา",
    "eBook Readers & Accessories": "เครื่องอ่าน e-Book และอุปกรณ์",
    "Cell Phones & Accessories": "โทรศัพท์มือถือและอุปกรณ์เสริม",
    "Accessories & Supplies": "อุปกรณ์เสริมและวัสดุ",
    "Video Projectors": "โปรเจกเตอร์",
    "Vehicle Electronics": "อิเล็กทรอนิกส์ยานพาหนะ",
    "Camera & Photo": "กล้องถ่ายรูปและการถ่ายภาพ",
    "Security & Surveillance Equipment": "อุปกรณ์รักษาความปลอดภัย",
    "Computers": "คอมพิวเตอร์",
    "Home Audio & Theater Products": "เสียงและโฮมเธียเตอร์",
    "Video Game Consoles & Accessories": "เครื่องเกมคอนโซลและอุปกรณ์",
    "Kids' Electronics": "อิเล็กทรอนิกส์สำหรับเด็ก",
    "Electronic Components": "ชิ้นส่วนอิเล็กทรอนิกส์",
    # Clothing & Fashion
    "Boys' Clothing": "เสื้อผ้าเด็กชาย",
    "Boys' Accessories": "อุปกรณ์เสริมเด็กชาย",
    "Boys' Jewelry": "เครื่องประดับเด็กชาย",
    "Boys' Watches": "นาฬิกาเด็กชาย",
    "Boys' Shoes": "รองเท้าเด็กชาย",
    "Boys' School Uniforms": "ชุดนักเรียนเด็กชาย",
    "Girls' Clothing": "เสื้อผ้าเด็กหญิง",
    "Girls' Accessories": "อุปกรณ์เสริมเด็กหญิง",
    "Girls' Jewelry": "เครื่องประดับเด็กหญิง",
    "Girls' Watches": "นาฬิกาเด็กหญิง",
    "Girls' Shoes": "รองเท้าเด็กหญิง",
    "Girls' School Uniforms": "ชุดนักเรียนเด็กหญิง",
    "Men's Clothing": "เสื้อผ้าผู้ชาย",
    "Men's Accessories": "อุปกรณ์เสริมผู้ชาย",
    "Men's Watches": "นาฬิกาผู้ชาย",
    "Men's Shoes": "รองเท้าผู้ชาย",
    "Women's Clothing": "เสื้อผ้าผู้หญิง",
    "Women's Handbags": "กระเป๋าถือผู้หญิง",
    "Women's Accessories": "อุปกรณ์เสริมผู้หญิง",
    "Women's Watches": "นาฬิกาผู้หญิง",
    "Women's Shoes": "รองเท้าผู้หญิง",
    "Women's Jewelry": "เครื่องประดับผู้หญิง",
    # Bags & Luggage
    "Travel Duffel Bags": "กระเป๋าเดินทางแบบ Duffel",
    "Messenger Bags": "กระเป๋าสะพายข้าง",
    "Travel Tote Bags": "กระเป๋าถือเดินทาง",
    "Garment Bags": "ถุงใส่เสื้อผ้า",
    "Luggage Sets": "ชุดกระเป๋าเดินทาง",
    "Suitcases": "กระเป๋าเดินทางล้อลาก",
    "Travel Accessories": "อุปกรณ์เดินทาง",
    "Rain Umbrellas": "ร่มกันฝน",
    "Backpacks": "เป้สะพายหลัง",
    "Luggage": "กระเป๋าเดินทาง",
    "Laptop Bags": "กระเป๋าแล็ปท็อป",
    # Health
    "Oral Care Products": "ผลิตภัณฑ์ดูแลช่องปาก",
    "Vision Products": "ผลิตภัณฑ์ดูแลสายตา",
    "Wellness & Relaxation Products": "สินค้าสุขภาพและผ่อนคลาย",
    "Household Supplies": "ของใช้ภายในบ้าน",
    "Health Care Products": "ผลิตภัณฑ์ดูแลสุขภาพ",
    "Diet & Sports Nutrition": "อาหารเสริมและโภชนาการกีฬา",
    "Home Use Medical Supplies & Equipment": "อุปกรณ์การแพทย์ใช้ที่บ้าน",
    "Sexual Wellness Products": "ผลิตภัณฑ์สุขภาพทางเพศ",
    "Health & Household": "สุขภาพและของใช้ในบ้าน",
    "Sports Nutrition Products": "ผลิตภัณฑ์โภชนาการกีฬา",
    # Home
    "Kids' Furniture": "เฟอร์นิเจอร์เด็ก",
    "Gift Cards": "บัตรของขวัญ",
    "Bath Products": "ผลิตภัณฑ์ห้องน้ำ",
    "Bedding": "ผ้าปูที่นอนและเครื่องนอน",
    "Home Décor Products": "ของตกแต่งบ้าน",
    "Furniture": "เฟอร์นิเจอร์",
    "Household Cleaning Supplies": "อุปกรณ์ทำความสะอาดบ้าน",
    "Seasonal Décor": "ของตกแต่งตามฤดูกาล",
    "Home Lighting & Ceiling Fans": "ไฟบ้านและพัดลมเพดาน",
    "Kitchen & Dining": "ครัวและรับประทานอาหาร",
    "Heating, Cooling & Air Quality": "ระบบทำความร้อน-เย็นและคุณภาพอากาศ",
    "Kids' Home Store": "สินค้าบ้านสำหรับเด็ก",
    "Home Storage & Organization": "การจัดเก็บและจัดระเบียบบ้าน",
    "Wall Art": "งานศิลปะติดผนัง",
    "Vacuum Cleaners & Floor Care": "เครื่องดูดฝุ่นและดูแลพื้น",
    "Ironing Products": "อุปกรณ์รีดผ้า",
    "Party Supplies": "อุปกรณ์จัดงานปาร์ตี้",
    "Home Appliances": "เครื่องใช้ไฟฟ้าภายในบ้าน",
    # Industrial
    "Stationery & Gift Wrapping Supplies": "เครื่องเขียนและอุปกรณ์ห่อของขวัญ",
    "Commercial Door Products": "ประตูเชิงพาณิชย์",
    "Power Transmission Products": "ผลิตภัณฑ์ส่งกำลัง",
    "Industrial Materials": "วัสดุอุตสาหกรรม",
    "Industrial Hardware": "ฮาร์ดแวร์อุตสาหกรรม",
    "Abrasive & Finishing Products": "ผลิตภัณฑ์ขัดและตกแต่งพื้นผิว",
    "Industrial Adhesives, Sealants & Lubricants": "กาว ซีลแลนท์และน้ำมันหล่อลื่นอุตสาหกรรม",
    "Material Handling Products": "อุปกรณ์จัดการวัสดุ",
    "Test, Measure & Inspect": "อุปกรณ์ทดสอบ วัดและตรวจสอบ",
    "Industrial Power & Hand Tools": "เครื่องมือไฟฟ้าและมืออุตสาหกรรม",
    "Hydraulics, Pneumatics & Plumbing": "ไฮดรอลิกส์ นิวแมติกส์และประปา",
    "Filtration": "ระบบกรอง",
    "Lab & Scientific Products": "อุปกรณ์ห้องปฏิบัติการและวิทยาศาสตร์",
    "Janitorial & Sanitation Supplies": "อุปกรณ์ทำความสะอาดและสุขาภิบาล",
    "Occupational Health & Safety Products": "อุปกรณ์อาชีวอนามัยและความปลอดภัย",
    "Cutting Tools": "เครื่องมือตัด",
    "Fasteners": "น็อต สลักเกลียวและอุปกรณ์ยึด",
    "Science Education Supplies": "อุปกรณ์การศึกษาด้านวิทยาศาสตร์",
    "Food Service Equipment & Supplies": "อุปกรณ์บริการอาหาร",
    "Additive Manufacturing Products": "ผลิตภัณฑ์การผลิตแบบเพิ่มเนื้อ (3D Printing)",
    "Professional Medical Supplies": "อุปกรณ์การแพทย์มืออาชีพ",
    "Professional Dental Supplies": "อุปกรณ์ทันตกรรมมืออาชีพ",
    "Packaging & Shipping Supplies": "อุปกรณ์บรรจุภัณฑ์และจัดส่ง",
    "Retail Store Fixtures & Equipment": "อุปกรณ์ร้านค้าปลีก",
    "Industrial & Scientific": "อุตสาหกรรมและวิทยาศาสตร์",
    # Pets
    "Pet Bird Supplies": "อุปกรณ์เลี้ยงนก",
    "Cat Supplies": "อุปกรณ์เลี้ยงแมว",
    "Dog Supplies": "อุปกรณ์เลี้ยงสุนัข",
    "Fish & Aquatic Pets": "ปลาและสัตว์น้ำ",
    "Horse Supplies": "อุปกรณ์เลี้ยงม้า",
    "Reptiles & Amphibian Supplies": "อุปกรณ์เลี้ยงสัตว์เลื้อยคลานและครึ่งบกน้ำ",
    "Small Animal Supplies": "อุปกรณ์เลี้ยงสัตว์เลี้ยงขนาดเล็ก",
    # Smart Home
    "Smart Home: New Smart Devices": "สมาร์ทโฮม: อุปกรณ์อัจฉริยะใหม่",
    "Smart Home: Voice Assistants and Hubs": "สมาร์ทโฮม: ลำโพงอัจฉริยะและฮับ",
    "Smart Home: Smart Locks and Entry": "สมาร์ทโฮม: ล็อคและระบบประตูอัจฉริยะ",
    "Smart Home: Home Entertainment": "สมาร์ทโฮม: ความบันเทิงในบ้าน",
    "Smart Home: WiFi and Networking": "สมาร์ทโฮม: WiFi และเครือข่าย",
    "Smart Home: Security Cameras and Systems": "สมาร์ทโฮม: กล้องวงจรปิดและระบบรักษาความปลอดภัย",
    "Smart Home: Lighting": "สมาร์ทโฮม: ไฟอัจฉริยะ",
    "Smart Home: Plugs and Outlets": "สมาร์ทโฮม: ปลั๊กและเต้ารับอัจฉริยะ",
    "Smart Home: Vacuums and Mops": "สมาร์ทโฮม: หุ่นยนต์ดูดฝุ่นและถูพื้น",
    "Smart Home Thermostats - Compatibility Checker": "สมาร์ทโฮม: เทอร์โมสตัทอัจฉริยะ",
    "Smart Home: Lawn and Garden": "สมาร์ทโฮม: สนามหญ้าและสวน",
    "Smart Home: Other Solutions": "สมาร์ทโฮม: โซลูชันอื่นๆ",
    "Smart Home - Heating & Cooling": "สมาร์ทโฮม: ระบบทำความร้อนและเย็น",
    # Sports
    "Sports & Fitness": "กีฬาและฟิตเนส",
    "Outdoor Recreation": "กิจกรรมกลางแจ้ง",
    "Sports & Outdoors": "กีฬาและกิจกรรมกลางแจ้ง",
    # Tools & Home Improvement
    "Pumps & Plumbing Equipment": "ปั๊มและอุปกรณ์ประปา",
    "Paint, Wall Treatments & Supplies": "สีและวัสดุทาผนัง",
    "Safety & Security": "ความปลอดภัยและการรักษาความปลอดภัย",
    "Light Bulbs": "หลอดไฟ",
    "Power Tools & Hand Tools": "เครื่องมือไฟฟ้าและเครื่องมือมือ",
    "Kitchen & Bath Fixtures": "อุปกรณ์ติดตั้งครัวและห้องน้ำ",
    "Lighting & Ceiling Fans": "ไฟและพัดลมเพดาน",
    "Electrical Equipment": "อุปกรณ์ไฟฟ้า",
    "Hardware": "ฮาร์ดแวร์และอุปกรณ์",
    "Building Supplies": "วัสดุก่อสร้าง",
    "Measuring & Layout": "เครื่องมือวัดและออกแบบ",
    "Welding & Soldering": "การเชื่อมและบัดกรี",
    "Tools & Home Improvement": "เครื่องมือและการปรับปรุงบ้าน",
    # Toys & Games
    "Kids' Party Supplies": "อุปกรณ์ปาร์ตี้เด็ก",
    "Toy Figures & Playsets": "ฟิกเกอร์และชุดของเล่น",
    "Novelty Toys & Amusements": "ของเล่นแปลกใหม่และความบันเทิง",
    "Building Toys": "ของเล่นต่อบล็อก",
    "Dolls & Accessories": "ตุ๊กตาและอุปกรณ์เสริม",
    "Games & Accessories": "เกมกระดานและอุปกรณ์",
    "Learning & Education Toys": "ของเล่นเพื่อการเรียนรู้และพัฒนาการ",
    "Kids' Dress Up & Pretend Play": "ของเล่นแต่งตัวและสวมบทบาท",
    "Puppets & Puppet Theaters": "หุ่นมือและโรงหุ่น",
    "Puzzles": "จิ๊กซอว์และปริศนา",
    "Sports & Outdoor Play Toys": "ของเล่นกีฬาและกลางแจ้ง",
    "Stuffed Animals & Plush Toys": "ตุ๊กตาผ้านุ่ม",
    "Tricycles, Scooters & Wagons": "รถสามล้อ สกู๊ตเตอร์และรถลาก",
    "Finger Toys": "ของเล่นนิ้วมือ",
    "Toy Vehicle Playsets": "ของเล่นยานพาหนะ",
    "Kids' Play Trains & Trams": "รถไฟของเล่น",
    "Kids' Play Trucks": "รถบรรทุกของเล่น",
    "Kids' Play Cars & Race Cars": "รถยนต์และรถแข่งของเล่น",
    "Kids' Play Boats": "เรือของเล่น",
    "Kids' Play Buses": "รถบัสของเล่น",
    "Kids' Play Tractors": "รถแทรกเตอร์ของเล่น",
    "Slot Cars, Race Tracks & Accessories": "สล็อตคาร์และสนามแข่ง",
    "Toys & Games": "ของเล่นและเกม",
    # Video Games
    "Sony PSP Games, Consoles & Accessories": "เกม Sony PSP",
    "Nintendo DS Games, Consoles & Accessories": "เกม Nintendo DS",
    "PlayStation 3 Games, Consoles & Accessories": "เกม PlayStation 3",
    "Wii Games, Consoles & Accessories": "เกม Wii",
    "Xbox 360 Games, Consoles & Accessories": "เกม Xbox 360",
    "Mac Games & Accessories": "เกม Mac",
    "Nintendo 3DS & 2DS Consoles, Games & Accessories": "เกม Nintendo 3DS & 2DS",
    "Legacy Systems": "เครื่องเกมรุ่นเก่า",
    "PlayStation Vita Games, Consoles & Accessories": "เกม PlayStation Vita",
    "Wii U Games, Consoles & Accessories": "เกม Wii U",
    "PlayStation 4 Games, Consoles & Accessories": "เกม PlayStation 4",
    "Xbox One Games, Consoles & Accessories": "เกม Xbox One",
    "Video Games": "วิดีโอเกม",
    "Online Video Game Services": "บริการเกมออนไลน์",
    "Virtual Reality Hardware & Accessories": "อุปกรณ์ VR และ Virtual Reality",
    "Nintendo Switch Consoles, Games & Accessories": "เกม Nintendo Switch",
    "PlayStation 5 Consoles, Games & Accessories": "เกม PlayStation 5",
    "Xbox Series X & S Consoles, Games & Accessories": "เกม Xbox Series X & S",
    "PC Games & Accessories": "เกม PC",
    "Unknown": "ไม่ระบุหมวดหมู่",
}

@st.cache_resource
def load_sales_model():
    mp = os.path.join(MODELS_DIR, 'sales_model.pkl')
    mm = os.path.join(MODELS_DIR, 'sales_model_meta.pkl')
    mc = os.path.join(MODELS_DIR, 'sales_category_map.pkl')
    if not os.path.exists(mp):
        return None, {}, []
    return (
        joblib.load(mp),
        joblib.load(mm) if os.path.exists(mm) else {},
        joblib.load(mc) if os.path.exists(mc) else []
    )

model, meta, cat_list = load_sales_model()

# Instruction panel
st.markdown("""
<div class="instruction-panel">
    <div class="instruction-title">วิธีใช้งาน</div>
    <ul class="instruction-steps">
        <li><span class="step-num">1</span>กรอกชื่อสินค้า (ไม่บังคับ — ใช้แสดงผลเท่านั้น ไม่กระทบผลทำนาย)</li>
        <li><span class="step-num">2</span>เลือกหมวดหมู่สินค้าจาก dropdown (category มีผลต่อการทำนาย)</li>
        <li><span class="step-num">3</span>ใส่ราคาขาย, ราคาตั้งต้น, คะแนนดาว และจำนวนรีวิว</li>
        <li><span class="step-num">4</span>กดปุ่ม "ทำนายยอดขาย" เพื่อดูผลลัพธ์</li>
        <li><span class="step-num">5</span>ดูผลทำนาย Low / Medium / High พร้อม probability แต่ละระดับ</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Layout
col_left, col_right = st.columns([1, 1.1], gap="large")

with col_left:
    st.markdown('<div class="panel"><div class="panel-title">ข้อมูลสินค้า</div>', unsafe_allow_html=True)

    product_name = st.text_input("ชื่อสินค้า (เพื่อแสดงผล)", placeholder="เช่น Sony ZV-1F Camera...")
    st.markdown('<div class="field-explain">ใส่หรือไม่ใส่ก็ได้ — ใช้แสดงชื่อในผลลัพธ์เท่านั้น</div>', unsafe_allow_html=True)

    category = st.selectbox(
        "หมวดหมู่สินค้า",
        cat_list if cat_list else ["Unknown"],
        format_func=lambda x: CATEGORY_TH.get(x, x)
    )
    st.markdown('<div class="field-explain"><b>สำคัญ:</b> แต่ละหมวดหมู่มีพฤติกรรมยอดขายต่างกัน เช่น อิเล็กทรอนิกส์มักมียอดขายสูงกว่าเครื่องประดับ</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        price_thb = st.number_input("ราคาขาย (บาท)", 1.0, 350_000.0, 1_049.0, 100.0, format="%.0f")
        st.markdown('<div class="field-explain">ราคาที่ลูกค้าจ่ายจริง (บาท) ช่วง 1–350,000 บาท</div>', unsafe_allow_html=True)
        stars = st.number_input("คะแนนดาว", 0.0, 5.0, 4.2, 0.1, format="%.1f")
        st.markdown('<div class="field-explain">ค่าเฉลี่ยดาวจากรีวิวลูกค้า ช่วง 0.0–5.0 ดาว</div>', unsafe_allow_html=True)
    with c2:
        list_price_thb = st.number_input("ราคาตั้งต้น (บาท)", 0.0, 350_000.0, 1_399.0, 100.0, format="%.0f")
        st.markdown('<div class="field-explain">ราคาก่อนลด ถ้าไม่มีส่วนลดให้ใส่เท่ากับราคาขาย</div>', unsafe_allow_html=True)
        reviews = st.number_input("จำนวนรีวิว", 0, 500_000, 1_200, 100)
        st.markdown('<div class="field-explain">จำนวนรีวิวทั้งหมด ยิ่งเยอะ = มีคนซื้อจริงมากกว่า</div>', unsafe_allow_html=True)

    discount_pct = max(0, (list_price_thb - price_thb) / max(list_price_thb, 0.01) * 100) if list_price_thb > 0 else 0
    if discount_pct > 0:
        st.info(f"ส่วนลด: **{discount_pct:.1f}%** (ประหยัด ฿{list_price_thb - price_thb:,.0f})")

    st.markdown("</div>", unsafe_allow_html=True)

    btn = st.button("📦 ทำนายยอดขาย", type="primary", use_container_width=True,
                    disabled=(model is None))
    if model is None:
        st.error("ไม่พบโมเดล — รัน `python train_sales_model.py` ก่อน")

with col_right:
    st.markdown('<div class="panel purple-top"><div class="panel-title">ผลการทำนาย</div>', unsafe_allow_html=True)

    if btn and model:
        try:
            price_usd      = price_thb     / USD_TO_THB
            list_price_usd = list_price_thb / USD_TO_THB
            log_r          = float(np.log1p(reviews))
            ppr            = price_thb / max(reviews, 1)
            vs             = stars * log_r
            rd             = reviews / max(price_thb, 1)

            row = {
                'stars':            stars,
                'reviews':          reviews,
                'price_thb':        price_thb,
                'listPrice_thb':    list_price_thb,
                'discount_pct':     discount_pct,
                'log_reviews':      log_r,
                'price_per_review': ppr,
                'value_score':      vs,
                'review_density':   rd,
                'cat_name':         category,
            }
            proba  = model.predict_proba(pd.DataFrame([row]))[0]   # [low, med, high]
            pred   = int(np.argmax(proba))
            labels = ['Low', 'Medium', 'High']
            ranges = ['0–50 ซื้อ/เดือน', '51–500 ซื้อ/เดือน', '>500 ซื้อ/เดือน']
            descs  = [
                'ยอดขายต่ำ — สินค้าอาจยังไม่เป็นที่นิยม หรืออยู่ใน niche ที่เล็กมาก ควรพิจารณาปรับราคาหรือ marketing',
                'ยอดขายปานกลาง — สินค้ามีฐานลูกค้า มีโอกาสเติบโตถ้าปรับกลยุทธ์ราคาหรือเพิ่มรีวิว',
                'ยอดขายสูง — สินค้าขายดีมาก มีความต้องการในตลาดสูง เหมาะสำหรับขยาย stock และ marketing',
            ]
            tier_cls = ['tier-low', 'tier-med', 'tier-high']
            lbl_cls  = ['tier-label-low', 'tier-label-med', 'tier-label-high']
            colors   = ['#F87171', '#FBBF24', '#34D399']
            icons    = ['🔴', '🟡', '🟢']

            name_display = product_name.strip() or CATEGORY_TH.get(category, category)
            st.markdown(f"""
            <div class="result-wrap {tier_cls[pred]}">
                <div class="result-label {lbl_cls[pred]}">Sales Volume Prediction — {name_display}</div>
                <div class="result-tier" style="color:{colors[pred]}">{icons[pred]} {labels[pred]}</div>
                <div class="result-sub" style="color:{colors[pred]}bb">{ranges[pred]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="panel" style="margin-top:.5rem; border-top-color:#718096">
                <div class="panel-title">คำแนะนำ</div>
                <div style="font-family:'Sarabun',sans-serif; font-size:.9rem; color:#4A5568; line-height:1.8">{descs[pred]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="panel" style="border-top-color:#718096"><div class="panel-title">ความน่าจะเป็นแต่ละระดับ</div>', unsafe_allow_html=True)
            bar_colors = ['#F87171', '#FBBF24', '#34D399']
            for i, (lbl, rng, p, c) in enumerate(zip(labels, ranges, proba, bar_colors)):
                pct = p * 100
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-header">
                        <span class="prob-label">{lbl} <span style="font-weight:400;color:#A0AEC0">({rng})</span></span>
                        <span class="prob-val" style="color:{c}">{pct:.1f}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{c}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Feature breakdown
            st.markdown('<div class="panel-title" style="margin-top:.5rem; font-family:\'Prompt\',sans-serif; font-size:.82rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#718096;">Features ที่ใช้ทำนาย</div>', unsafe_allow_html=True)
            details = [
                ("⭐ Stars",           f"{stars:.1f}"),
                ("📝 Reviews",         f"{reviews:,}"),
                ("💰 ราคาขาย",         f"฿{price_thb:,.0f}"),
                ("🏷️ ราคาตั้งต้น",    f"฿{list_price_thb:,.0f}"),
                ("🔻 ส่วนลด",          f"{discount_pct:.1f}%"),
                ("📊 log_reviews",     f"{log_r:.3f}"),
                ("💡 value_score",     f"{vs:.2f}"),
                ("📐 price/review",    f"฿{ppr:.2f}"),
                ("🔢 review_density",  f"{rd:.4f}"),
            ]
            cols = st.columns(3)
            for i, (k, v) in enumerate(details):
                with cols[i % 3]:
                    st.metric(k, v)

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
    else:
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">📦</div>
            <div class="placeholder-text">
                ใส่ข้อมูลสินค้าทางซ้าย<br>
                แล้วกด <b style="color:#E85D04">ทำนายยอดขาย</b><br><br>
                <span style="font-size:.85rem; color:#475569">
                    โมเดลจะทำนายว่าสินค้านี้มียอดขาย<br>
                    ระดับ Low / Medium / High ต่อเดือน
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Tier explanation panel below
st.markdown("""
<div class="panel" style="margin-top:1.5rem; border-top-color:#718096">
    <div class="panel-title">ความหมายของ Low / Medium / High</div>
    <div class="tier-explain-grid">
        <div class="tier-explain-card">
            <div class="tier-explain-badge teb-high">High</div>
            <div class="tier-explain-range">มากกว่า 500 ซื้อ/เดือน</div>
            <div class="tier-explain-desc">สินค้าขายดีมาก มีฐานลูกค้ากว้าง ความต้องการสูง เหมาะลงทุนและขยาย</div>
        </div>
        <div class="tier-explain-card">
            <div class="tier-explain-badge teb-med">Medium</div>
            <div class="tier-explain-range">51 – 500 ซื้อ/เดือน</div>
            <div class="tier-explain-desc">ยอดขายปานกลาง มีฐานลูกค้า โอกาสเติบโตถ้าปรับกลยุทธ์</div>
        </div>
        <div class="tier-explain-card">
            <div class="tier-explain-badge teb-low">Low</div>
            <div class="tier-explain-range">0 – 50 ซื้อ/เดือน</div>
            <div class="tier-explain-desc">ยอดขายต่ำ อาจเป็น niche product หรือยังไม่เป็นที่รู้จัก</div>
        </div>
    </div>
    <div style="margin-top:1rem; font-family:'Sarabun',sans-serif; font-size:.82rem; color:#718096; line-height:1.7; padding:.8rem 1rem; background:#F7FAFC; border-radius:10px; border:1px solid #2D3250;">
        <strong style="color:#4A5568;">หมายเหตุ:</strong>
        ผลทำนายอิงจากข้อมูล Amazon Products Dataset 2023 และพฤติกรรมการซื้อในช่วงนั้น
        ตัวเลข "ซื้อ/เดือน" มาจากฟีเจอร์ <code>boughtInLastMonth</code> ใน dataset
        ผลทำนายเป็นการประมาณการ ไม่ใช่การรับประกัน
    </div>
</div>
""", unsafe_allow_html=True)
