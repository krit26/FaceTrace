FROM python:3.9

ENV TF_USE_LEGACY_KERAS=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    KMP_DUPLICATE_LIB_OK=TRUE

RUN apt-get -y update \
    && apt-get install -y libgl1-mesa-glx

RUN mkdir -p /FaceTrace

COPY ./components /FaceTrace/components
COPY ./configs /FaceTrace/configs
COPY ./configurations /FaceTrace/configurations
COPY ./constants /FaceTrace/constants
COPY ./models /FaceTrace/models
COPY ./stores /FaceTrace/stores
COPY ./structures /FaceTrace/structures
COPY ./utils /FaceTrace/utils
COPY ./weights /FaceTrace/weights
COPY ./playground /FaceTrace/playground
COPY ./requirements.txt /FaceTrace/requirements.txt
COPY ./app.py /FaceTrace/app.py
COPY ./README.md /FaceTrace/README.md

RUN pip install -r /FaceTrace/requirements.txt
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu --index-url https://download.pytorch.org/whl/cpu

ENV PYTHONPATH=/FaceTrace
WORKDIR /FaceTrace