FROM svizor/zoomcamp-model:3.11.5-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model.bin", "dv.bin", "./"]

EXPOSE 9698

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9698", "predict:app"]