import dagshub
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/sweety-mahale/MLOps_Project.mlflow")
dagshub.init(repo_owner='sweety-mahale', repo_name='MLOps_Project', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)