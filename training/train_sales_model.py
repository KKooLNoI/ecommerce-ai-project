"""
train_sales_model.py
─────────────────────────────────────────────────────────────────
เทรน ML Ensemble ทำนายยอดขาย (Sales Volume) ของสินค้า Amazon

Input  : data/amazon_products.csv + data/amazon_categories.csv
Output : models/sales_model.pkl
         models/sales_model_meta.pkl
         models/sales_category_map.pkl

Target : boughtInLastMonth → 3 ระดับ
         Low    (0–50   ซื้อ/เดือน)
         Medium (51–500 ซื้อ/เดือน)
         High   (>500   ซื้อ/เดือน)

Features:
  Numeric  : stars, reviews, price_thb, listPrice_thb, discount_pct,
             log_reviews, price_per_review, value_score, review_density
  Category : category_name (OneHotEncoder)

โมเดล:
  RandomForest + XGBoost + HistGBM → VotingClassifier(soft)

อัตราแลกเปลี่ยน: 1 USD = 35 THB
─────────────────────────────────────────────────────────────────
"""

import sys
import pandas as pd
import numpy as np
import os
import time
import multiprocessing

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from xgboost import XGBClassifier

MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

USD_TO_THB = 35.0   # อัตราแลกเปลี่ยน 1 USD = 35 THB
N_CPU = multiprocessing.cpu_count()
print(f"[CPU] {N_CPU} cores")

# ── XGBoost device ────────────────────────────────────────────
def get_xgb_device():
    try:
        import xgboost as xgb
        p = xgb.XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
        p.fit([[1]], [0])
        print("[XGB] GPU (CUDA)")
        return 'cuda'
    except Exception:
        print("[XGB] CPU (hist)")
        return 'cpu'

XGB_DEVICE = get_xgb_device()

# ═══════════════════════════════════════════════════════
# 1. โหลดและ Merge ข้อมูล
# ═══════════════════════════════════════════════════════
print("\n[1] โหลดข้อมูล...")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, 'dataset', 'ML', 'amazon_products.csv'), encoding='utf-8')
cats = pd.read_csv(os.path.join(ROOT, 'dataset', 'ML', 'amazon_categories.csv'), encoding='utf-8')
print(f"    products  : {len(df):,} แถว")
print(f"    categories: {len(cats):,} แถว")

# Merge category_name เข้า products
df = df.merge(cats.rename(columns={'id': 'category_id', 'category_name': 'cat_name'}),
              on='category_id', how='left')
df['cat_name'] = df['cat_name'].fillna('Unknown')

# ═══════════════════════════════════════════════════════
# 2. Clean + Feature Engineering
# ═══════════════════════════════════════════════════════
print("\n[2] Cleaning & Feature Engineering...")

def clean_num(s):
    return pd.to_numeric(
        s.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce'
    )

df['price']             = clean_num(df['price'])
df['listPrice']         = clean_num(df['listPrice'])
df['stars']             = pd.to_numeric(df['stars'],   errors='coerce')
df['reviews']           = pd.to_numeric(df['reviews'], errors='coerce')
df['boughtInLastMonth'] = pd.to_numeric(df['boughtInLastMonth'], errors='coerce').fillna(0)

# แปลงราคาเป็นบาท
df['price_thb']     = df['price']     * USD_TO_THB
df['listPrice_thb'] = df['listPrice'] * USD_TO_THB

# Feature Engineering
df['discount_pct']     = np.where(
    (df['listPrice'] > 0) & df['listPrice'].notna() & (df['price'] > 0),
    ((df['listPrice'] - df['price']) / df['listPrice'] * 100).clip(0, 99),
    0.0
)
df['log_reviews']      = np.log1p(df['reviews'].fillna(0))
df['price_per_review'] = np.where(
    df['reviews'].fillna(0) > 0,
    df['price_thb'].fillna(0) / df['reviews'],
    df['price_thb'].fillna(0)
)
df['value_score']      = df['stars'].fillna(0) * df['log_reviews']
df['review_density']   = df['reviews'].fillna(0) / df['price_thb'].fillna(1).clip(lower=1)

# กรอง outlier
df = df[
    (df['price_thb'].fillna(0) >= 1) &
    (df['price_thb'].fillna(0) <= 350_000) &
    (df['stars'].fillna(0).between(0, 5))
].copy()

print(f"    ข้อมูลหลังกรอง: {len(df):,} แถว")

# ═══════════════════════════════════════════════════════
# 3. สร้าง Target — 3 ระดับยอดขาย
# ═══════════════════════════════════════════════════════
def sales_tier(n):
    if n <= 50:  return 0   # Low
    if n <= 500: return 1   # Medium
    return 2                # High

df['sales_label'] = df['boughtInLastMonth'].apply(sales_tier)

counts = df['sales_label'].value_counts().sort_index()
labels = ['Low (0-50)', 'Medium (51-500)', 'High (>500)']
for i, lbl in enumerate(labels):
    n = counts.get(i, 0)
    print(f"    {lbl:<20} : {n:>7,} ({n/len(df)*100:.1f}%)")

# บันทึก category map
sales_category_map = sorted(df['cat_name'].unique().tolist())
joblib.dump(sales_category_map, os.path.join(MODELS_DIR, 'sales_category_map.pkl'))
print(f"\n    หมวดหมู่ทั้งหมด: {len(sales_category_map)} รายการ")

# ═══════════════════════════════════════════════════════
# 4. Features & Target
# ═══════════════════════════════════════════════════════
numeric_features     = ['stars', 'reviews', 'price_thb', 'listPrice_thb',
                         'discount_pct', 'log_reviews', 'price_per_review',
                         'value_score', 'review_density']
categorical_features = ['cat_name']

X = df[numeric_features + categorical_features].copy()
y = df['sales_label'].copy()

# Sample ถ้าข้อมูลเกิน 500k
if len(X) > 500_000:
    print(f"\n[Sample] ลดจาก {len(X):,} → 500,000 แถว (stratified)")
    _, idx = train_test_split(range(len(X)), test_size=500_000/len(X),
                              stratify=y, random_state=42)
    X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    print(f"    Low:{(y==0).sum():,}  Med:{(y==1).sum():,}  High:{(y==2).sum():,}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ═══════════════════════════════════════════════════════
# 5. Preprocessing Pipeline
# ═══════════════════════════════════════════════════════
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_features),
    ('cat', cat_pipe, categorical_features)
])

# ═══════════════════════════════════════════════════════
# 6. Ensemble — RF + XGBoost + HistGBM
# ═══════════════════════════════════════════════════════
print("\n[6] สร้าง Ensemble Classifier...")

m1 = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
m2 = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='multi:softprob',
    num_class=3,
    tree_method='hist',
    device=XGB_DEVICE,
    n_jobs=-1,
    random_state=42,
    verbosity=0,
    eval_metric='mlogloss'
)
m3 = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    l2_regularization=0.1,
    class_weight='balanced',
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[('rf', m1), ('xgb', m2), ('hgb', m3)],
    voting='soft',
    n_jobs=-1
)

clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   ensemble)
])

# ═══════════════════════════════════════════════════════
# 7. เทรน
# ═══════════════════════════════════════════════════════
print(f"\n[7] เทรน Ensemble (RF + XGB + HGB)...")
t0 = time.time()
clf.fit(X_train, y_train)
elapsed = time.time() - t0
print(f"    เสร็จใน {elapsed:.1f} วินาที")

# ═══════════════════════════════════════════════════════
# 8. ประเมินผล
# ═══════════════════════════════════════════════════════
y_pred  = clf.predict(X_test)
acc     = accuracy_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred, average='weighted')
f1_mac  = f1_score(y_test, y_pred, average='macro')

print(f"\n[8] ผลการประเมิน:")
print(f"    Accuracy       : {acc*100:.2f}%")
print(f"    F1 (weighted)  : {f1:.4f}")
print(f"    F1 (macro)     : {f1_mac:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Low','Medium','High'])}")

# ═══════════════════════════════════════════════════════
# 9. บันทึกโมเดล
# ═══════════════════════════════════════════════════════
joblib.dump(clf, os.path.join(MODELS_DIR, 'sales_model.pkl'))

meta = {
    'accuracy':              round(acc,     4),
    'f1_weighted':           round(f1,      4),
    'f1_macro':              round(f1_mac,  4),
    'train_seconds':         round(elapsed, 1),
    'xgb_device':            XGB_DEVICE,
    'n_cpu':                 N_CPU,
    'numeric_features':      numeric_features,
    'categorical_features':  categorical_features,
    'usd_to_thb':            USD_TO_THB,
    'n_train':               int(len(X_train)),
    'n_test':                int(len(X_test)),
    'class_labels':          ['Low', 'Medium', 'High'],
    'thresholds':            {'low': 50, 'medium': 500},
}
joblib.dump(meta, os.path.join(MODELS_DIR, 'sales_model_meta.pkl'))

print(f"\n[OK] models/sales_model.pkl")
print(f"[OK] models/sales_model_meta.pkl")
print(f"[OK] models/sales_category_map.pkl")
print(f"\n{meta}")
