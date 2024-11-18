FROM gcr.io/kaggle-gpu-images/python@sha256:940180dcea42e703acbeae669a52d68e6fa3130ef60805b36005534a5152a5f3

USER root
RUN apt install git --yes
RUN pip install ruff
RUN pip install pytest
RUN pip install pre-commit

# competition specific
RUN pip install ultralytics
RUN pip install einops
RUN pip install bitsandbytes