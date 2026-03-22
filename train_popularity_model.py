"""
train_popularity_model.py
─────────────────────────────────────────────────────────────────
เทรน ML Ensemble เพื่อทำนายว่า Amazon product จะเป็น Best Seller ไหม

Input  : data/amazon_products.csv  (Kaggle: asaniczka/amazon-products-dataset-2023)
Output : models/popularity_model.pkl
         models/popularity_model_meta.pkl
         models/category_map.pkl

Target : isBestSeller (binary 0/1)
         → predict_proba → "buy_score" 0–100%

ฟีเจอร์:
  Numeric  : stars, reviews, price, listPrice, discount_pct,
             price_per_review, review_density, log_reviews
  Category : main_category (OneHotEncoder)

โมเดล:
  RF + XGBoost(GPU) + HistGBM → VotingClassifier(soft)
─────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import os
import time
import multiprocessing

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from xgboost import XGBClassifier

os.makedirs('models', exist_ok=True)

N_CPU = multiprocessing.cpu_count()
print(f"🖥️  CPU cores : {N_CPU}")

def get_xgb_device():
    try:
        import xgboost as xgb
        p = xgb.XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
        p.fit([[1]], [0])
        print("🎮 XGBoost : GPU (CUDA) ✅")
        return 'cuda'
    except Exception:
        print("🖥️  XGBoost : CPU (hist)")
        return 'cpu'

XGB_DEVICE = get_xgb_device()

# ═══════════════════════════════════════════════════════
# 1. โหลดข้อมูล
# ═══════════════════════════════════════════════════════
print("\n📂 กำลังโหลด amazon_products.csv ...")
try:
    df = pd.read_csv('data/amazon_products.csv', encoding='utf-8')
    print(f"✅ โหลดสำเร็จ : {len(df):,} แถว, {len(df.columns)} คอลัมน์")
    print(f"   คอลัมน์ : {list(df.columns)}")
except Exception as e:
    print(f"❌ โหลดไม่ได้: {e}")
    exit(1)

# ═══════════════════════════════════════════════════════
# 2. Clean + Feature Engineering
# ═══════════════════════════════════════════════════════
print("\n🔧 Feature Engineering...")

# แปลง isBestSeller → int
df['isBestSeller'] = df['isBestSeller'].astype(str).str.strip().str.lower()
df['isBestSeller'] = df['isBestSeller'].map({'true': 1, 'false': 0, '1': 1, '0': 0})
df = df.dropna(subset=['isBestSeller'])
df['isBestSeller'] = df['isBestSeller'].astype(int)

# แปลง numeric columns
def clean_price(s):
    return pd.to_numeric(
        s.astype(str).str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    )

if 'price' in df.columns:    df['price']     = clean_price(df['price'])
if 'listPrice' in df.columns: df['listPrice'] = clean_price(df['listPrice'])
if 'stars'    in df.columns:  df['stars']     = pd.to_numeric(df['stars'],   errors='coerce')
if 'reviews'  in df.columns:  df['reviews']   = pd.to_numeric(df['reviews'], errors='coerce')

# Feature Engineering
df['discount_pct']    = np.where(
    (df['listPrice'] > 0) & df['listPrice'].notna() & (df['price'] > 0),
    ((df['listPrice'] - df['price']) / df['listPrice'] * 100).clip(0, 99),
    0.0
)
df['log_reviews']     = np.log1p(df['reviews'].fillna(0))
df['price_per_review']= np.where(
    df['reviews'].fillna(0) > 0,
    df['price'].fillna(0) / df['reviews'],
    df['price'].fillna(0)
)
df['value_score']     = df['stars'].fillna(0) * df['log_reviews']
df['review_density']  = df['reviews'].fillna(0) / (df['price'].fillna(1).clip(lower=1))

# หมวดหมู่หลัก
cat_col = None
for c in ['main_category', 'category', 'category_id']:
    if c in df.columns:
        cat_col = c
        break

if cat_col is None:
    df['main_category'] = 'unknown'
    cat_col = 'main_category'

df[cat_col] = df[cat_col].astype(str).str.strip().fillna('unknown')

# กรอง outlier
df = df[
    (df['price'].fillna(0) >= 0.01) &
    (df['price'].fillna(0) <= 10000) &
    (df['stars'].fillna(0) >= 0) &
    (df['stars'].fillna(0) <= 5)
].copy()

print(f"   ข้อมูลหลังกรอง  : {len(df):,} แถว")
print(f"   Best Seller     : {df['isBestSeller'].sum():,} ({df['isBestSeller'].mean()*100:.1f}%)")
print(f"   Not Best Seller : {(1-df['isBestSeller']).sum():,} ({(1-df['isBestSeller']).mean()*100:.1f}%)")

# บันทึก category mapping ไว้ให้ app ใช้
category_map = {v: i for i, v in enumerate(sorted(df[cat_col].unique()))}
joblib.dump(category_map, 'models/category_map.pkl')
print(f"   หมวดหมู่        : {len(category_map)} category")

# ═══════════════════════════════════════════════════════
# 3. Features & Target
# ═══════════════════════════════════════════════════════
numeric_features     = ['stars', 'reviews', 'price', 'listPrice',
                         'discount_pct', 'log_reviews', 'price_per_review',
                         'value_score', 'review_density']
categorical_features = [cat_col]

X = df[numeric_features + categorical_features].copy()
y = df['isBestSeller'].copy()

# sample ถ้าข้อมูลเกิน 500k (เพื่อให้เทรนเร็ว)
if len(X) > 500_000:
    print(f"\n⚡ Sample 500,000 แถวจาก {len(X):,} (stratified)")
    idx = np.concatenate([
        np.random.choice(np.where(y == 0)[0], 460_000, replace=False),
        np.random.choice(np.where(y == 1)[0], min(40_000, (y==1).sum()), replace=False)
    ])
    X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    print(f"   หลัง sample : {len(X):,} แถว | Best Seller : {y.sum():,} ({y.mean()*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n   Train : {len(X_train):,} | Test : {len(X_test):,}")

# ═══════════════════════════════════════════════════════
# 4. Preprocessing Pipeline
# ═══════════════════════════════════════════════════════
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_features),
    ('cat', cat_pipe, categorical_features)
])

# ═══════════════════════════════════════════════════════
# 5. Ensemble Classifier
#
#   class_weight/scale_pos_weight รับมือ imbalanced dataset
# ═══════════════════════════════════════════════════════
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / max(pos, 1)
print(f"\n🔧 scale_pos_weight = {scale:.1f}")

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
    scale_pos_weight=scale,
    tree_method='hist',
    device=XGB_DEVICE,
    n_jobs=-1,
    random_state=42,
    verbosity=0,
    eval_metric='auc'
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
# 6. เทรน
# ═══════════════════════════════════════════════════════
print(f"\n⏳ กำลังเทรน Ensemble Classifier (RF + XGB + HGB)...")
t0 = time.time()
clf.fit(X_train, y_train)
elapsed = time.time() - t0
print(f"⏱️  เทรนเสร็จใน {elapsed:.1f} วินาที")

# ═══════════════════════════════════════════════════════
# 7. ประเมินผล
# ═══════════════════════════════════════════════════════
y_pred      = clf.predict(X_test)
y_proba     = clf.predict_proba(X_test)[:, 1]

acc         = accuracy_score(y_test, y_pred)
f1          = f1_score(y_test, y_pred, average='weighted')
roc_auc     = roc_auc_score(y_test, y_proba)

print(f"\n📊 ผลการประเมินโมเดล:")
print(f"   Accuracy  : {acc*100:.2f}%")
print(f"   F1 Score  : {f1:.4f}")
print(f"   ROC-AUC   : {roc_auc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Not BestSeller','BestSeller'])}")

# ═══════════════════════════════════════════════════════
# 8. บันทึกโมเดล + metadata
# ═══════════════════════════════════════════════════════
joblib.dump(clf, 'models/popularity_model.pkl')

meta = {
    'accuracy':       round(acc,    4),
    'f1_score':       round(f1,     4),
    'roc_auc':        round(roc_auc,4),
    'train_seconds':  round(elapsed, 1),
    'xgb_device':     XGB_DEVICE,
    'n_cpu':          N_CPU,
    'numeric_features':     numeric_features,
    'categorical_features': categorical_features,
    'cat_col':        cat_col,
    'n_train':        int(len(X_train)),
    'n_test':         int(len(X_test)),
    'best_seller_pct':round(float(y.mean()) * 100, 2),
}
joblib.dump(meta, 'models/popularity_model_meta.pkl')

print(f"\n✅ เซฟโมเดลที่    models/popularity_model.pkl")
print(f"✅ เซฟ metadata ที่ models/popularity_model_meta.pkl")
print(f"✅ เซฟ category ที่ models/category_map.pkl")
print(f"\n{meta}")
