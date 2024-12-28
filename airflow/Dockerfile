FROM apache/airflow:2.10.4

COPY requirements_airflow.txt .

RUN python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple innowise-ml-internship-alexennk==1.0.5

RUN pip install -r requirements_airflow.txt

RUN export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YmY3NjQ0ZC0wMzM1LTQ2MjAtYTE4Ny0wZmE2MmYyZWI1ZTUifQ=="

RUN export NEPTUNE_PROJECT="alexennk/innowise-neptune"