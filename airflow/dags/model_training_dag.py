import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator



with DAG(
    dag_id = 'model_training',
    start_date = pendulum.datetime(
        2026, 8, 11,
        tz= 'Asia/Tehran'
    ),
    schedule="*/15 * * * *",
    catchup= False,
    max_active_runs = 1,
    tags = ['ml','training']

) as dag:
    run_model = BashOperator(
        task_id = 'run_model',
        bash_command="python -u /opt/airflow/src/model_train.py",
        retries = 2,
        retry_delay = pendulum.duration(minutes=5),
    )

    register_model = BashOperator(
    task_id='register_model',
    bash_command="python -u /opt/airflow/src/model_register.py",
)
    run_model >> register_model