#!/usr/bin/env python3
"""古文字识别 - 改进版"""
import os, json, glob
import cv2
import numpy as np
from paddleocr import PaddleOCR

def preprocess_image(img_path):
    """预处理：加强对比度 + 降噪 + 锐化"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # CLAE对比度增强
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    # 降噪
    img = cv2.fastNlMeansDenoising(img, h=10)
    # 锐化
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    # 转回三通道（PaddleOCR需要）
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def run_inference(input_dir='/saisdata/13/eval/images/', output_file='/saisresult/prediction.json'):
    ocr = PaddleOCR(
        use_angle_cls=True, lang='ch', show_log=False, use_gpu=True,
        gpu_mem=8000,
        det_db_thresh=0.2,       # 调低：让更多候选框出来
        det_db_box_thresh=0.3,   # 调低：别太严格
        det_db_unclip_ratio=2.0, # 放大检测框
        rec_batch_num=6
    )

    images = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    results = {}

    for img_path in images:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        try:
            # 先用预处理过的图试试
            preprocessed = preprocess_image(img_path)
            if preprocessed is not None:
                result = ocr.ocr(preprocessed, cls=False)
            else:
                result = ocr.ocr(img_path, cls=False)

            chars = []
            if result and result[0]:
                for line in result[0]:
                    chars.append({'text': line[1][0], 'confidence': line[1][1]})
            results[img_id] = chars
        except:
            results[img_id] = []

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    total = sum(len(v) for v in results.values())
    print(f"[INFO] 处理完成: {len(results)} 张图, 检测到 {total} 个字符")

if __name__ == '__main__':
    run_inference()
