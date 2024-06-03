FROM tensorflow/tensorflow:2.16.1-gpu-jupyter
LABEL authors="hubert"

WORKDIR /app

RUN pip install --no-cache-dir pandas==2.2.2 tensorflow-io==0.37.0 librosa==0.10.2 matplotlib==3.8.3

RUN apt install nano

COPY . .

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--allow-root"]

