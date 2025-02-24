FROM nvcr.io/nvidia/rapidsai/base:23.12-cuda12.0-py3.10
USER root
# Install commands
RUN apt-get update && apt-get install -y git
RUN apt-get install -y vim
RUN apt-get install -y build-essential
RUN apt-get install -y wget

#COPY requirements.txt /tmp/requirements.txt
#RUN pip install --no-cache-dir --requirement /requirements.txt

RUN pip install --no-cache-dir --root-user-action=ignore -U  tqdm jsonlines 

RUN pip install --no-cache-dir --root-user-action=ignore -U numpy transformers vllm autoawq

