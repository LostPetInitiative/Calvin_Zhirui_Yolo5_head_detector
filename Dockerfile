# downloader stage is used to obtain the model code from provate repo + model weights from Zenodo
FROM ubuntu AS downloader
WORKDIR /work
RUN apt-get update && apt-get install --no-install-recommends -y ca-certificates git wget

# we MUST use multistage build here to avoid storing PAT in image history
ARG GITHUB_USER
ARG GITHUB_PAT
RUN mkdir /app
# copying the YoloV5
RUN git clone --depth=1 https://$GITHUB_USER:$GITHUB_PAT@github.com/LostPetInitiative/study_spring_2022.git
RUN cp -r study_spring_2022/zhirui/yolov5 /app/yolov5
# copying the YoloV5 weights
RUN wget https://zenodo.org/record/6663662/files/yolov5s.pt -O /app/yolov5s.pt

FROM python:3.9-slim AS FINAL

# installing openCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /requirements.txt

# --extra-index-url https://download.pytorch.org/whl/cpu avoids CUDA installation
RUN python -m pip install --upgrade pip && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /requirements.txt
COPY --from=downloader /app .

ENV KAFKA_URL=kafka:9092
ENV INPUT_QUEUE=kashtanka_distinct_photos_pet_cards
ENV OUTPUT_QUEUE=kashtanka_calvin_zhirui_yolov5_output
CMD python3 serve.py
COPY code .

FROM FINAL as TESTS
COPY example /app/example
RUN python -m unittest discover -v
