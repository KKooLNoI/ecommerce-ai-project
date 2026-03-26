"""
train_image_model.py
─────────────────────────────────────────────────────────────────
เทรน CNN (EfficientNetB0) จำแนกหมวดหมู่สินค้า Amazon จากรูปภาพ

Input  : data/images/<category>/  (Kaggle: ahmedelsayedrashad/amazon-products-image)
Output : models/product_image_model.keras
         models/product_image_model_best.keras
         models/product_image_model_classes.json

Architecture:
  EfficientNetB0 (pretrained ImageNet) + custom head
  Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Softmax

Training:
  Phase 1 (lr=1e-3): head only, 20 epochs
  Phase 2 (lr=1e-5): fine-tune top 50 layers, 30 epochs
─────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import numpy as np
import multiprocessing
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# รองรับ Windows terminal ที่ใช้ cp1252 (ไม่รองรับ emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── GPU / CPU setup ──────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Mixed precision: float16 compute, float32 weights → เร็วขึ้น ~2× บน GPU
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"[GPU] {len(gpus)} GPU(s) — mixed_float16 enabled")
else:
    N_CPU = multiprocessing.cpu_count()
    # intra = N_CPU (parallelism ภายใน op), inter = 2 (ป้องกัน thread contention)
    tf.config.threading.set_intra_op_parallelism_threads(N_CPU)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    print(f"[CPU] {N_CPU} cores (intra={N_CPU}, inter=2)")

# XLA JIT: compile ops ให้เร็วขึ้น ~10-20% บน CPU
tf.config.optimizer.set_jit(True)

# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════
IMG_SIZE   = (160, 160)   # 160 balance ระหว่างความเร็วและ accuracy (96 เล็กเกิน)
BATCH_SIZE = 64           # 128 → 64: gradient ดีขึ้น
AUTOTUNE   = tf.data.AUTOTUNE
DATA_DIR   = os.path.join(ROOT, 'dataset', 'CNN', 'images')

# 500 steps × 64 = 32,000 img/epoch — มากพอให้โมเดล converge
STEPS_PER_EPOCH = 500
VAL_STEPS       = 100

# ═══════════════════════════════════════════════════════
# 1. ตรวจสอบ dataset — สร้าง folder ถ้ายังไม่มี
# ═══════════════════════════════════════════════════════
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

def count_images(folder):
    # os.scandir เร็วกว่า rglob ~5× สำหรับ flat directory
    try:
        return sum(1 for e in os.scandir(folder)
                   if e.is_file() and os.path.splitext(e.name)[1].lower() in IMG_EXTS)
    except OSError:
        return 0

def find_dataset():
    """ค้นหา dataset อัตโนมัติในหลาย path ที่เป็นไปได้"""
    candidates = [
        DATA_DIR,                          # data/images/
        os.path.join(ROOT, 'dataset', 'CNN', 'amazon-products-image'),
        os.path.join(ROOT, 'dataset', 'CNN', 'Amazon Products Image'),
        os.path.join(ROOT, 'dataset', 'CNN', 'images', 'train'),
    ]
    for path in candidates:
        if os.path.exists(path):
            cats = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if cats:
                return path, cats
    return None, []

# สร้าง data/images/ ถ้ายังไม่มี
os.makedirs(DATA_DIR, exist_ok=True)

found_dir, categories = find_dataset()

if not found_dir:
    print("=" * 60)
    print("  ไม่พบรูปภาพในระบบ")
    print("=" * 60)
    print()
    print("  วิธีดาวน์โหลด dataset:")
    print()
    print("  1) ใช้ Kaggle CLI (แนะนำ):")
    print("     kaggle datasets download -d ahmedelsayedrashad/amazon-products-image --unzip -p data/images/")
    print()
    print("  2) ดาวน์โหลดมือ:")
    print("     → kaggle.com/datasets/ahmedelsayedrashad/amazon-products-image")
    print("     → แตกไฟล์ ZIP แล้ววางใน e:\\AI\\ecommerce-project\\data\\images\\")
    print()
    print("  โครงสร้าง folder ที่ต้องการ:")
    print("     data/images/")
    print("     ├── Electronics/")
    print("     │   ├── img1.jpg")
    print("     │   └── ...")
    print("     ├── Clothing/")
    print("     └── ...")
    print()
    print("  [OK] folder data/images/ ถูกสร้างให้แล้ว — วางรูปได้เลย")
    exit(0)

# ถ้าเจอใน path อื่น → อัปเดต DATA_DIR
if found_dir != DATA_DIR:
    print(f"  พบ dataset ใน '{found_dir}' → ใช้ path นี้แทน")
    DATA_DIR = found_dir

print(f"[OK] พบ {len(categories)} หมวดหมู่ใน {DATA_DIR}")

# กรอง category ที่มีรูปน้อยเกิน (< 20 รูป) ออก
MIN_IMAGES = 20
valid_cats = []
for cat in categories:
    n = count_images(os.path.join(DATA_DIR, cat))
    if n >= MIN_IMAGES:
        valid_cats.append((cat, n))
        print(f"   [OK] {cat:<30} {n:>6,} รูป")
    else:
        print(f"   [WARN]  ข้าม '{cat}' ({n} รูป < {MIN_IMAGES})")

if not valid_cats:
    print()
    print("=" * 60)
    print("  ไม่มี category ที่มีรูปเพียงพอ (ต้องการอย่างน้อย 20 รูป/category)")
    print("=" * 60)
    print()
    print(f"  folder ที่พบ : {DATA_DIR}")
    print(f"  category ที่พบ : {categories}")
    print()
    print("  ตรวจสอบว่าแตกไฟล์ถูกต้อง:")
    print("  โครงสร้างที่ถูกต้อง:")
    print(f"    {DATA_DIR}/")
    print(f"    ├── <ชื่อ category>/")
    print(f"    │   ├── image1.jpg")
    print(f"    │   └── ...")
    print(f"    └── ...")
    exit(0)

print(f"\n   ใช้ {len(valid_cats)} category:")
for c, n in sorted(valid_cats, key=lambda x: -x[1]):
    print(f"   {c:<30} {n:>6,} รูป")

# รายชื่อ category ที่ผ่านการกรองแล้ว — ใช้ส่งให้ make_dataset
VALID_CLASS_NAMES = sorted([cat for cat, _ in valid_cats])
# counts_dict สำหรับ class_weights — ใช้จากที่นับไปแล้ว ไม่ต้อง scan disk ซ้ำ
CAT_COUNTS = {cat: n for cat, n in valid_cats}


# ═══════════════════════════════════════════════════════
# 2. tf.data Pipeline
#
#   [WARN]  EfficientNetB3 มี Rescaling(1/255) + Normalization อยู่ภายในตัวเองแล้ว
#      → ห้าม normalize ก่อนส่งเข้าโมเดล มิฉะนั้น pixel จะกลายเป็น ~0.004
#      → ส่งค่า raw float32 [0, 255] เข้า base model โดยตรง
#
#   class_names_filter → กรองเฉพาะ category ที่ผ่าน MIN_IMAGES check
# ═══════════════════════════════════════════════════════
def make_dataset(data_dir, subset, augment=False, class_names_filter=None):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        class_names=class_names_filter,   # ← กรอง category ที่มีรูปน้อยออก
        validation_split=0.2,
        subset=subset,
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=(subset == 'training')
    )
    class_names = ds.class_names
    num_classes = len(class_names)

    # raw [0-255] → cast float32 เท่านั้น (EfficientNetB3 จัดการ rescale เอง)
    augment_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
    ])

    to_float = lambda x, y: (tf.cast(x, tf.float32), y)

    # ไม่ใช้ .cache() — dataset 295k ไฟล์ ใช้ RAM ~58GB+ ถ้า cache → swap → ช้ามาก
    if augment:
        ds = ds.map(to_float, num_parallel_calls=AUTOTUNE).map(
            lambda x, y: (augment_layer(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    else:
        ds = ds.map(to_float, num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE), class_names, num_classes


# ═══════════════════════════════════════════════════════
# 3. Class Weights (handle imbalanced categories)
# ═══════════════════════════════════════════════════════
def get_class_weights(class_names, counts_dict):
    """รับ counts_dict={cat: n} ที่นับไปแล้ว — ไม่ต้อง scan disk ซ้ำ"""
    counts = np.array([max(1, counts_dict.get(c, 1)) for c in class_names], dtype=np.float64)
    total  = counts.sum()
    n_cls  = len(class_names)
    return {i: total / (n_cls * c) for i, c in enumerate(counts)}


# ═══════════════════════════════════════════════════════
# 4. Build Model — MobileNetV2
# ═══════════════════════════════════════════════════════
def build_model(num_classes):
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False

    inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=Adam(1e-3),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model, base


# ═══════════════════════════════════════════════════════
# 5. เทรน
# ═══════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"  เทรน CNN: product_image_model  (MobileNetV2)")
print(f"{'='*55}")

train_ds, class_names, num_classes = make_dataset(DATA_DIR, 'training',  augment=True,  class_names_filter=VALID_CLASS_NAMES)
val_ds,   _,           _           = make_dataset(DATA_DIR, 'validation', augment=False, class_names_filter=VALID_CLASS_NAMES)
class_weights = get_class_weights(class_names, CAT_COUNTS)

print(f"  {num_classes} classes : {class_names}")
model, base = build_model(num_classes)

CKPT = os.path.join(MODELS_DIR, 'product_image_model_best.keras')

# ── Phase 1 ──────────────────────────────────────────────
print("\n  Phase 1 : เทรน head (base frozen, lr=1e-3)...")
h1 = model.fit(
    train_ds, epochs=20,
    steps_per_epoch=STEPS_PER_EPOCH,   # 500×64 = 32,000 img/epoch
    validation_data=val_ds,
    validation_steps=VAL_STEPS,        # 100×64 = 6,400 img/val
    class_weight=class_weights,
    callbacks=[
        EarlyStopping('val_accuracy', patience=6, min_delta=0.001,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(CKPT, monitor='val_accuracy', save_best_only=True, verbose=0),
        ReduceLROnPlateau('val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
)

# ── Phase 2 ──────────────────────────────────────────────
print("\n  Phase 2 : fine-tune top 30 layers (lr=1e-5)...")
base.trainable = True
for layer in base.layers[:-30]:   # MobileNetV2 มี ~155 layers → 30 เพียงพอ
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)
h2 = model.fit(
    train_ds, epochs=25,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=VAL_STEPS,
    class_weight=class_weights,
    callbacks=[
        EarlyStopping('val_accuracy', patience=7, min_delta=0.001,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(CKPT, monitor='val_accuracy', save_best_only=True, verbose=0),
        ReduceLROnPlateau('val_loss', factor=0.3, patience=4, min_lr=1e-8, verbose=1)
    ]
)

best_acc = max(h1.history['val_accuracy'] + h2.history['val_accuracy'])
print(f"\n  [OK] Best val_accuracy : {best_acc*100:.1f}%")

model.save(os.path.join(MODELS_DIR, 'product_image_model.keras'))

# บันทึก class indices → JSON
class_indices = {name: i for i, name in enumerate(class_names)}
with open(os.path.join(MODELS_DIR, 'product_image_model_classes.json'), 'w', encoding='utf-8') as f:
    json.dump(class_indices, f, ensure_ascii=False, indent=2)

print(f"\n[OK] models/product_image_model.keras")
print(f"[OK] models/product_image_model_best.keras")
print(f"[OK] models/product_image_model_classes.json")
print(f"   Class map : {class_indices}")
