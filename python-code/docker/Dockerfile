FROM python:3.6

COPY ./requirements.txt /requirements.txt
# COPY ./models /root/.keras/models/
# WORKDIR /app

RUN pip --version
RUN python --version

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

COPY ./custom_lib/Lib/site-packages /usr/local/lib/python3.6/site-packages

# CMD ["cat", "/usr/local/lib/python3.6/site-packages/tensorflow/python/keras/layers/advanced_activations.py"]

# CMD ["python", "main.py"]
