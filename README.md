# 🛒 E-Commerce Product Review AI

ระบบ AI วิเคราะห์สินค้า Amazon 2 มุมมอง:
- **ML Ensemble** (Random Forest + XGBoost + HistGBM) ทำนายยอดขายจากข้อมูลตัวเลข
- **CNN** (MobileNetV2 Transfer Learning) จำแนกหมวดหมู่สินค้าจากรูปภาพ

🌐 **Demo:** https://ecommerce-ai-project.streamlit.app

---

## 📁 โครงสร้าง Repository

```
ecommerce-ai-project/
│
├── webapp/                        ← โค้ด Web Application (Streamlit)
│   ├── Home.py                    ← หน้าแรก
│   └── pages/
│       ├── 1_Data_Overview.py     ← EDA และ Visualization
│       ├── 2_ML_Model_Info.py     ← อธิบาย ML Ensemble
│       ├── 3_CNN_Model_Info.py    ← อธิบาย CNN Model
│       ├── 5_Test_ML.py           ← ทดสอบ ML Model
│       └── 6_Test_CNN.py          ← ทดสอบ CNN Model
│
├── training/                      ← โค้ด Train Model
│   ├── train_sales_model.py       ← เทรน ML Ensemble (RF+XGB+HistGBM)
│   └── train_image_model.py       ← เทรน CNN (MobileNetV2)
│
├── dataset/                       ← Dataset
│   ├── ML/                        ← Structured Data (Amazon Products)
│   │   ├── amazon_products_sample.csv   ← ตัวอย่าง 50,000 แถว
│   │   └── amazon_categories.csv        ← รายการหมวดหมู่ทั้งหมด
│   └── CNN/                       ← Image Data (Amazon Product Images)
│       └── sample_images/         ← ตัวอย่างรูปภาพ 20 รูป/หมวดหมู่
│           ├── Electronics_and_Appliances/
│           ├── Fashion/
│           ├── Household_Essentials/
│           ├── accessories/
│           ├── other/
│           ├── sports_fitness/
│           └── stores/
│
├── models/                        ← Trained Models (output)
│   ├── sales_model.pkl
│   ├── sales_model_meta.pkl
│   ├── sales_category_map.pkl
│   ├── product_image_model_best.keras
│   └── product_image_model_classes.json
│
├── requirements.txt
└── How_to_run.txt                 ← คู่มือการรันโปรเจกต์
```

---

## 📊 Dataset

### Dataset 1 — ML (Structured Data)
| รายละเอียด | ข้อมูล |
|-----------|--------|
| ชื่อ | Amazon Products Dataset 2023 |
| ที่มา | [Kaggle — asaniczka](https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products) |
| ขนาด | 1,426,337 แถว (359 MB) |
| Features | ราคา, คะแนนดาว, จำนวนรีวิว, หมวดหมู่ |
| Label | `boughtInLastMonth` → Low / Medium / High |

> **หมายเหตุ:** ไฟล์ฉบับเต็ม (`amazon_products.csv`) มีขนาดใหญ่เกิน GitHub
> ใน repository มี sample 50,000 แถว (`amazon_products_sample.csv`)
> ดาวน์โหลดฉบับเต็มได้ที่ Kaggle ด้านบน

### Dataset 2 — CNN (Image Data)
| รายละเอียด | ข้อมูล |
|-----------|--------|
| ชื่อ | Amazon Products Image Dataset |
| ที่มา | [Kaggle — ahmedelsayedrashad](https://www.kaggle.com/datasets/ahmedelsayedrashad/amazon-products-image) |
| ขนาด | 295,431 รูปภาพ (7 GB) |
| หมวดหมู่ | 7 categories (Electronics, Fashion, Household, ...) |

> **หมายเหตุ:** รูปภาพฉบับเต็ม (7 GB) ใหญ่เกิน GitHub
> ใน repository มีตัวอย่าง 20 รูป/หมวดหมู่ใน `dataset/CNN/sample_images/`
> ดาวน์โหลดฉบับเต็มได้ที่ Kaggle ด้านบน

---

## 🚀 วิธีรัน

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. ดาวน์โหลด Dataset (ฉบับเต็ม)
```bash
# ML Dataset
kaggle datasets download -d asaniczka/amazon-products-dataset-2023-1-4m-products --unzip -p dataset/ML/

# CNN Dataset
kaggle datasets download -d ahmedelsayedrashad/amazon-products-image --unzip -p dataset/CNN/images/
```

### 3. เทรน Models
```bash
# ML Ensemble
python training/train_sales_model.py

# CNN
python training/train_image_model.py
```

### 4. รัน Web App
```bash
streamlit run webapp/Home.py
```

---

## 🤖 Models

### ML Ensemble (Structured Data)
- **Random Forest** — ต้านทาน noise ดี
- **XGBoost** — จัดการ imbalanced data
- **HistGradientBoosting** — เร็ว ใช้ RAM น้อย
- Soft Voting รวมผล 3 โมเดล

### CNN (Image Classification)
- **MobileNetV2** pretrained บน ImageNet
- Transfer Learning 2-Phase
- Input: 160×160 px | Categories: 7 หมวดหมู่

---

## 📚 References
- asaniczka (2023). *Amazon Products Dataset 2023*. Kaggle.
- Rashad, A. E. (2023). *Amazon Products Image*. Kaggle.
- Sandler et al. (2018). MobileNetV2. *CVPR 2018*.
