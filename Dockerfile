FROM registry.cn-hangzhou.aliyuncs.com/sais/competition:base-paddle-cuda11.2-cudnn8-ubuntu18.04
WORKDIR /workspace
COPY infer.py /workspace/infer.py
RUN pip install paddleocr==2.8.1 paddlepaddle-gpu==2.6.1 -i https://mirror.baidu.com/pypi/simple
CMD ["python3", "/workspace/infer.py"]
