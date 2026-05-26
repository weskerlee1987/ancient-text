#!/usr/bin/env python3
import os, json, glob, sys
from paddleocr import PaddleOCR

def run_inference(input_dir='/saisdata/13/eval/images/', output_file='/saisresult/prediction.json'):
    ocr = PaddleOCR(use_angle_cls=False, lang='ch', show_log=False, use_gpu=True, gpu_mem=8000, det_db_thresh=0.3, det_db_box_thresh=0.5, rec_batch_num=6)
    images = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    results = {}
    for img_path in images:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        try:
            result = ocr.ocr(img_path, cls=False)
            chars = [{'text': line[1][0], 'confidence': line[1][1]} for line in (result[0] if result and result[0] else [])]
            results[img_id] = chars
        except:
            results[img_id] = []
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 完成！{len(results)} 张图, {sum(len(v) for v in results.values())} 个字符")

if __name__ == '__main__':
    run_inference()
