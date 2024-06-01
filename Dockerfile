FROM tensorflow/tensorflow:latest-gpu-jupyter
LABEL authors="hubert"

WORKDIR /app

RUN pip install --no-cache-dir pandas tensorflow-io librosa matplotlib

RUN apt install nano

COPY . .

EXPOSE 8888

ENV LANGUAGES=""


CMD ["/app/scripts/Spoken-language-detection/set_envs.sh"]

#CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--allow-root"]

