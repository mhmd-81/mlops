FROM apache/airflow:3.3.0

COPY ../requirements.txt /requirements.txt

RUN pip install --no-cache-dir \
-i https://package-mirror.liara.ir/repository/pypi/simple \
-r requirements.txt

