FROM paddlepaddle/paddle:2.6.1-gpu-cuda11.7-cudnn8.4-trt8.4

RUN pip install paddleocr==2.8.1 opencv-python Pillow tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace
COPY infer.py /workspace/infer.py

CMD ["python3", "/workspace/infer.py"]
