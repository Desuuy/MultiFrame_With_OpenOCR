import os
import sys

# Thiết lập sys.path giống train_rec.py để import được "tools.xxx"
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.create_lmdb_dataset import get_datalist
# createDataset gốc không dùng nữa, ta tự viết bản nhỏ hơn với map_size nhỏ.

# === CẤU HÌNH RIÊNG CHO BIỂN SỐ CỦA BẠN ===
data_dir = r"C:\Users\anhhu\lp_for_openocr\images"        # thư mục ảnh
label_file = r"C:\Users\anhhu\lp_for_openocr\labels.txt"  # file labels.txt
save_path = r"C:\Users\anhhu\lp_for_openocr\lmdb"         # nơi lưu LMDB
max_len = 25  # độ dài text tối đa

print("data_dir  :", data_dir)
print("label_file:", label_file)
print("save_path :", save_path)

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
    for imagePath, label in tqdm(data_list,
                                 desc=f'make dataset, save to {outputPath}'):
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


# === PHẦN MAIN (ở ngoài hàm) ===
train_data_list = get_datalist(data_dir, label_file, max_len)
print("num samples in label list:", len(train_data_list))

createDataset_small(train_data_list, save_path)
print("Done.")