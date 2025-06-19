FROM python:3.12-slim

ENV VIRTUAL_ENV=/opt/venv
WORKDIR /

COPY Simple-Naive-Bayes-Multinomial.py /
COPY dataClickId/ /dataClickId/
COPY combined_clickid.csv /
COPY naive_bayes_model.csv /
COPY predictions.csv /

# RUN pip install pandas scikit-learn <- the simple version doesn't even use scikit-learn and pandas

# COPY app/ /app/
CMD ["python", "Simple-Naive-Bayes-Multinomial.py"]

# Labeling image
LABEL maintainer="Timotius Darrel Putra Arma <darrel2208@gmail.com>"
LABEL version="1.0"
LABEL description="Simple (Built-in) and Complex (external) Naive Bayes Multinomial for classify and predict clickbait news"
