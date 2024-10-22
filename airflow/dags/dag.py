from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG('train_model_dag', default_args=default_args, schedule_interval='@daily')

train_model_task = BashOperator(
    task_id='train_model',
    bash_command='docker run -p 5000:5000 my_rnn_model',
    dag=dag,
)
