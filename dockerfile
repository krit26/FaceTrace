FROM python:3.9

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
COPY ./requirements.txt /FaceTrace/requirements.txt
COPY ./app.py /FaceTrace/app.py
COPY ./README.md /FaceTrace/README.md

RUN pip install -r /FaceTrace/requirements.txt

ENV PYTHONPATH=/FaceTrace
WORKDIR /FaceTrace