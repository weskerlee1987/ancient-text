#!/usr/bin/env python3
"""极简推理 - 提交保底"""
import os, sys, json, glob
from paddleocr import PaddleOCR

INPUT_DIR = '/saisdata/13/eval/images/'
OUTPUT_FILE = '/saisresult/prediction.json'

def main():
    print(f"[INFO] 输入: {INPUT_DIR}")
    print(f"[INFO] 输出: {OUTPUT_FILE}")
    
    ocr = PaddleOCR(use_angle_cls=False, lang='ch', show_log=False, use_gpu=True)
    
    images = sorted(glob.glob(os.path.join(INPUT_DIR, '*.png')))
    if not images:
        images = sorted(glob.glob(os.path.join(INPUT_DIR, '*')))
    print(f"[INFO] 图片数: {len(images)}")
    
    results = {}
    for img_path in images:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        try:
            result = ocr.ocr(img_path, cls=False)
            chars = []
            if result and result[0]:
                for line in result[0]:
                    pts = line[0]
                    text, conf = line[1]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    chars.append({
                        "bbox": [int(min(xs)), int(min(ys)), 
                                 int(max(xs)-min(xs)), int(max(ys)-min(ys))],
                        "text": text
                    })
            results[img_id] = chars
        except:
            results[img_id] = []
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
    
    total = sum(len(v) for v in results.values())
    print(f"[INFO] 完成! {len(results)}张, {total}字符")

if __name__ == '__main__':
    main()
