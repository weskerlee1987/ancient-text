FROM paddlepaddle/paddle:2.6.1-gpu-cuda11.7-cudnn8.4-trt8.4

RUN pip install paddleocr==2.8.1 opencv-python Pillow tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app
COPY infer.py /app/infer.py

RUN echo '#!/bin/bash\npython3 /app/infer.py' > /app/run.sh && chmod +x /app/run.sh

CMD ["bash", "/app/run.sh"]
