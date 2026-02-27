import os
import sys

# Thiết lập sys.path giống train_rec.py để import được "tools.xxx"
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.create_lmdb_dataset import get_datalist
# createDataset gốc không dùng nữa, ta tự viết bản nhỏ hơn với map_size nhỏ.

# ============================================================
# CẤU HÌNH DATASET CHO TRAIN VÀ VALID (RIÊNG BIỆT)
# ============================================================
# Mỗi phần tử là một dict với:
#   - data_dir: thư mục gốc chứa ảnh
#   - label_file: file labels.txt tương ứng (đường dẫn tuyệt đối hoặc tương đối)

TRAIN_DATASETS = [
    {
        "data_dir": r"C:\Users\anhhu\lmdb_for_openocr\train_images",
        "label_file": r"C:\Users\anhhu\lmdb_for_openocr\train_labels.txt",
    },
    # Có thể thêm nhiều source train khác ở đây nếu muốn gộp:
    # {
    #     "data_dir": r"C:\Users\anhhu\lmdb_for_openocr\train_lr_images",
    #     "label_file": r"C:\Users\anhhu\lmdb_for_openocr\train_lr_labels.txt",
    # },
]

VAL_DATASETS = [
    {
        "data_dir": r"C:\Users\anhhu\lmdb_for_openocr\val_images",
        "label_file": r"C:\Users\anhhu\lmdb_for_openocr\val_labels.txt",
    },
    # Có thể thêm nhiều source valid khác ở đây nếu cần.
]

# Thư mục lưu LMDB cho train / valid (2 thư mục khác nhau)
TRAIN_SAVE_PATH = r"D:\lmdb_data_for_openocr\lmdb_train"
VAL_SAVE_PATH = r"D:\lmdb_data_for_openocr\lmdb_val"

max_len = 25  # độ dài text tối đa

print("TRAIN_SAVE_PATH :", TRAIN_SAVE_PATH)
print("VAL_SAVE_PATH   :", VAL_SAVE_PATH)

import lmdb
import cv2
import numpy as np
import io
from PIL import Image
from tqdm import tqdm


def createDataset_small(data_list, outputPath, checkValid=True, map_size=8 * 1024**3):
    """Bản rút gọn của createDataset với map_size nhỏ hơn (mặc định ~8GB)."""
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=map_size)
    cache = {}
    cnt = 1
    for imagePath, label in tqdm(
            data_list, desc=f'make dataset, save to {outputPath}'):
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            buf = io.BytesIO(imageBin)
            w, h = Image.open(buf).size
        if checkValid:
            try:
                imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
                img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
                imgH, imgW = img.shape[0], img.shape[1]
                if imgH * imgW == 0:
                    print(f'{imagePath} is not a valid image')
                    continue
            except:
                continue

        imageKey = f'image-{cnt:09d}'.encode()
        labelKey = f'label-{cnt:09d}'.encode()
        whKey = f'wh-{cnt:09d}'.encode()
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[whKey] = (f'{w}_{h}').encode()

        if cnt % 1000 == 0:
            with env.begin(write=True) as txn:
                for k, v in cache.items():
                    txn.put(k, v)
            cache = {}
        cnt += 1

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)
    print('Created dataset with %d samples' % nSamples)


def build_and_save_lmdb(tag, dataset_cfgs, save_path):
    """Gộp nhiều source (data_dir, label_file) rồi tạo 1 LMDB."""
    print(f"\n===== BUILD {tag.upper()} LMDB =====")
    print("save_path:", save_path)

    all_data = []
    for cfg in dataset_cfgs:
        data_dir = cfg["data_dir"]
        label_file = cfg["label_file"]
        print("data_dir  :", data_dir)
        print("label_file:", label_file)
        if not os.path.isfile(label_file):
            print("  -> BỎ QUA: file không tồn tại. Tạo file này hoặc sửa đường dẫn.")
            continue
        if not os.path.isdir(data_dir):
            print("  -> BỎ QUA: thư mục ảnh không tồn tại.")
            continue
        data_list = get_datalist(data_dir, label_file, max_len)
        print("  -> num samples from this dataset:", len(data_list))
        all_data.extend(data_list)

    print(f"TOTAL samples for {tag}:", len(all_data))
    if not all_data:
        print(f"Lỗi: {tag} không có mẫu nào, bỏ qua tạo LMDB cho phần này.")
        return

    createDataset_small(all_data, save_path)
    print(f"{tag} LMDB Done.")


# === PHẦN MAIN (ở ngoài hàm) ===
if __name__ == "__main__":
    build_and_save_lmdb("train", TRAIN_DATASETS, TRAIN_SAVE_PATH)
    build_and_save_lmdb("valid", VAL_DATASETS, VAL_SAVE_PATH)