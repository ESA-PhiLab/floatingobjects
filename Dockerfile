from pytorch/pytorch:latest

COPY requirements.txt .
RUN conda install -c conda-forge gdal
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y git
RUN python -m pip install -U git+https://github.com/qubvel/segmentation_models.pytorch

COPY code/ .
