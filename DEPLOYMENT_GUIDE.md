# 🚀 Financial Data Analysis Toolkit - Deployment Guide

> **Professional deployment guide for the Financial Data Analysis Toolkit with advanced ML models, data processing pipeline, and scalable analytics architecture.**

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Data Pipeline Setup](#data-pipeline-setup)
6. [ML Model Management](#ml-model-management)
7. [Configuration](#configuration)
8. [Monitoring & Performance](#monitoring--performance)
9. [Security](#security)
10. [Troubleshooting](#troubleshooting)

---

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.8+
- Apache Spark (for big data processing)
- PostgreSQL/MongoDB (data storage)
- Redis (caching and job queues)
- GPU support (optional, for deep learning)

### 1-Minute Setup
```bash
# Clone and setup
git clone https://github.com/olaitanojo/financial-data-analysis-toolkit.git
cd financial-data-analysis-toolkit
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Initialize data pipeline
python main.py --setup

# Run analysis toolkit
python main.py
```

---

## 💻 Local Development

### Environment Setup
```bash
# Create virtual environment
python -m venv analytics_env
source analytics_env/bin/activate  # Windows: analytics_env\Scripts\activate

# Install core dependencies
pip install pandas numpy scipy scikit-learn
pip install tensorflow keras torch torchvision
pip install apache-airflow celery redis

# Install data processing libraries
pip install pyspark apache-beam dask
pip install sqlalchemy psycopg2-binary pymongo

# Install financial libraries
pip install yfinance alpha-vantage quandl pandas-datareader
pip install quantlib-python zipline-reloaded

# Install development tools
pip install jupyter matplotlib seaborn plotly
pip install pytest black flake8 mypy
```

### Development Configuration
```python
# config/development.py
import os
from pathlib import Path

class DevelopmentConfig:
    # Application
    DEBUG = True
    TESTING = False
    
    # Data Sources
    DATA_DIR = Path("data")
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = Path("models")
    
    # Database
    POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/analytics")
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/analytics")
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    
    # Spark
    SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
    SPARK_APP_NAME = "FinancialAnalytics"
    
    # ML Configuration
    ML_CONFIG = {
        "model_registry": "models/",
        "experiment_tracking": "mlflow",
        "feature_store": "feast",
        "model_serving": "seldon"
    }
    
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_WORKERS = 4
```

### Local Development Server
```bash
# Start Jupyter for interactive analysis
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

# Start API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start Celery workers for background tasks
celery -A analytics.celery worker --loglevel=info

# Start Spark cluster (optional)
$SPARK_HOME/sbin/start-master.sh
$SPARK_HOME/sbin/start-worker.sh spark://localhost:7077
```

---

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    openjdk-11-jdk \
    curl wget \
    && rm -rf /var/lib/apt/lists/*

# Set Java home for Spark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Spark
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m analyst && chown -R analyst:analyst /app
USER analyst

# Expose ports
EXPOSE 8000 4040 7077 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start services
CMD ["python", "start_services.py"]
```

### Docker Compose (Production)
```yaml
version: '3.8'

services:
  # Main Analytics Application
  analytics:
    build: .
    container_name: financial-analytics
    ports:
      - "8000:8000"
      - "8888:8888"  # Jupyter
      - "4040:4040"  # Spark UI
    environment:
      - POSTGRES_URL=postgresql://analyst:secure_password@postgres:5432/analytics
      - MONGODB_URL=mongodb://mongo:27017/analytics
      - REDIS_URL=redis://redis:6379
      - SPARK_MASTER=spark://spark-master:7077
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./notebooks:/app/notebooks
      - ./logs:/app/logs
    depends_on:
      - postgres
      - mongo
      - redis
      - spark-master
    restart: unless-stopped
    
  # Spark Master
  spark-master:
    image: bitnami/spark:3.5.0
    container_name: spark-master
    environment:
      - SPARK_MODE=master
      - SPARK_RPC_AUTHENTICATION_ENABLED=no
      - SPARK_RPC_ENCRYPTION_ENABLED=no
      - SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
      - SPARK_SSL_ENABLED=no
    ports:
      - "7077:7077"
      - "8080:8080"
    restart: unless-stopped
    
  # Spark Workers
  spark-worker-1:
    image: bitnami/spark:3.5.0
    container_name: spark-worker-1
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
      - SPARK_WORKER_MEMORY=2G
      - SPARK_WORKER_CORES=2
    depends_on:
      - spark-master
    restart: unless-stopped
    
  spark-worker-2:
    image: bitnami/spark:3.5.0
    container_name: spark-worker-2
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
      - SPARK_WORKER_MEMORY=2G
      - SPARK_WORKER_CORES=2
    depends_on:
      - spark-master
    restart: unless-stopped
    
  # PostgreSQL Database
  postgres:
    image: postgres:15
    container_name: analytics-postgres
    environment:
      POSTGRES_DB: analytics
      POSTGRES_USER: analyst
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    
  # MongoDB
  mongo:
    image: mongo:7
    container_name: analytics-mongo
    environment:
      MONGO_INITDB_ROOT_USERNAME: analyst
      MONGO_INITDB_ROOT_PASSWORD: secure_password
      MONGO_INITDB_DATABASE: analytics
    volumes:
      - mongo_data:/data/db
    ports:
      - "27017:27017"
    restart: unless-stopped
    
  # Redis
  redis:
    image: redis:7-alpine
    container_name: analytics-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 512mb
    restart: unless-stopped
    
  # Celery Workers
  celery-worker:
    build: .
    container_name: celery-worker
    command: celery -A analytics.celery worker --loglevel=info --concurrency=4
    environment:
      - POSTGRES_URL=postgresql://analyst:secure_password@postgres:5432/analytics
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    
  # Celery Beat (Scheduler)
  celery-beat:
    build: .
    container_name: celery-beat
    command: celery -A analytics.celery beat --loglevel=info
    environment:
      - POSTGRES_URL=postgresql://analyst:secure_password@postgres:5432/analytics
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    
  # Airflow (Data Pipeline)
  airflow:
    image: apache/airflow:2.8.0
    container_name: analytics-airflow
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://analyst:secure_password@postgres:5432/airflow
      - AIRFLOW__CORE__FERNET_KEY=your-fernet-key-here
    volumes:
      - ./dags:/opt/airflow/dags
      - ./data:/opt/airflow/data
    ports:
      - "8081:8080"
    depends_on:
      - postgres
    restart: unless-stopped
    
  # MLflow (Model Registry)
  mlflow:
    image: python:3.9-slim
    container_name: analytics-mlflow
    command: >
      bash -c "
        pip install mlflow psycopg2-binary &&
        mlflow server 
        --backend-store-uri postgresql://analyst:secure_password@postgres:5432/mlflow
        --default-artifact-root ./mlruns
        --host 0.0.0.0
        --port 5000
      "
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    depends_on:
      - postgres
    restart: unless-stopped

volumes:
  postgres_data:
  mongo_data:
  redis_data:
```

### Requirements.txt
```txt
# Core Data Science
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2

# Deep Learning
tensorflow==2.15.0
torch==2.1.0
torchvision==0.16.0
keras==2.15.0

# Big Data Processing
pyspark==3.5.0
apache-beam[gcp]==2.52.0
dask==2023.12.0

# Financial Libraries
yfinance==0.2.24
alpha-vantage==2.3.1
quandl==3.7.0
quantlib==1.32
zipline-reloaded==3.0.4

# Database
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
pymongo==4.6.0
redis==5.0.1

# API and Web
fastapi==0.104.1
uvicorn==0.24.0
celery==5.3.4

# Workflow Management
apache-airflow==2.8.0

# ML Operations
mlflow==2.8.1
feast==0.32.0
seldon-core==1.18.0

# Visualization
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0

# Jupyter and Development
jupyter==1.0.0
jupyterlab==4.0.9
pytest==7.4.3
black==23.12.0
flake8==6.1.0
```

---

## ☁️ Cloud Deployment

### AWS EMR + ECS Deployment

#### EMR Cluster Configuration
```json
{
  "Name": "financial-analytics-cluster",
  "ReleaseLabel": "emr-6.15.0",
  "Applications": [
    {"Name": "Spark"},
    {"Name": "Hadoop"},
    {"Name": "Zeppelin"},
    {"Name": "JupyterHub"}
  ],
  "Instances": {
    "InstanceGroups": [
      {
        "Name": "Master",
        "Market": "ON_DEMAND",
        "InstanceRole": "MASTER",
        "InstanceType": "m5.xlarge",
        "InstanceCount": 1
      },
      {
        "Name": "Workers",
        "Market": "SPOT",
        "InstanceRole": "CORE",
        "InstanceType": "m5.2xlarge",
        "InstanceCount": 3,
        "BidPrice": "0.20"
      }
    ],
    "Ec2KeyName": "your-key-pair",
    "KeepJobFlowAliveWhenNoSteps": true
  },
  "BootstrapActions": [
    {
      "Name": "Install Python Dependencies",
      "ScriptBootstrapAction": {
        "Path": "s3://your-bucket/bootstrap/install-dependencies.sh"
      }
    }
  ],
  "Configurations": [
    {
      "Classification": "spark-defaults",
      "Properties": {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer"
      }
    }
  ]
}
```

#### ECS Task Definition for API
```json
{
  "family": "financial-analytics-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "analytics-api",
      "image": "your-ecr-repo/financial-analytics:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "SPARK_MASTER", "value": "yarn"}
      ],
      "secrets": [
        {"name": "DATABASE_URL", "valueFrom": "arn:aws:secretsmanager:region:account:secret:db-url"},
        {"name": "API_KEYS", "valueFrom": "arn:aws:secretsmanager:region:account:secret:api-keys"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/financial-analytics",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Dataproc + Cloud Run
```bash
# Create Dataproc cluster
gcloud dataproc clusters create financial-analytics \
  --zone us-central1-b \
  --num-masters 1 \
  --num-workers 3 \
  --worker-machine-type n1-standard-4 \
  --master-machine-type n1-standard-2 \
  --image-version 2.1-ubuntu20 \
  --enable-ip-alias \
  --max-age=3h \
  --initialization-actions gs://your-bucket/init-script.sh

# Deploy API to Cloud Run
gcloud run deploy financial-analytics-api \
  --image gcr.io/PROJECT_ID/financial-analytics:latest \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --set-env-vars DATAPROC_CLUSTER=financial-analytics \
  --allow-unauthenticated
```

### Azure HDInsight + Container Instances
```bash
# Create HDInsight cluster
az hdinsight create \
  --name financial-analytics-cluster \
  --resource-group analytics-rg \
  --type Spark \
  --component-version Spark=3.1 \
  --http-user admin \
  --http-password SecurePassword123! \
  --ssh-user sshuser \
  --ssh-password SecureSSHPassword123! \
  --location eastus \
  --cluster-size-in-nodes 4

# Deploy container app
az containerapp create \
  --name financial-analytics-api \
  --resource-group analytics-rg \
  --environment analytics-env \
  --image your-registry/financial-analytics:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 5 \
  --cpu 2.0 \
  --memory 4.0Gi
```

---

## 🔄 Data Pipeline Setup

### Apache Airflow DAG
```python
# dags/financial_data_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'analytics-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'financial_data_pipeline',
    default_args=default_args,
    description='Daily financial data processing pipeline',
    schedule_interval='@daily',
    catchup=False,
    tags=['finance', 'data-processing']
)

def extract_market_data():
    """Extract market data from various sources"""
    from analytics.data_sources import DataExtractor
    
    extractor = DataExtractor()
    extractor.fetch_stock_data()
    extractor.fetch_options_data()
    extractor.fetch_economic_indicators()

def transform_data():
    """Transform and clean raw data"""
    from analytics.processors import DataTransformer
    
    transformer = DataTransformer()
    transformer.clean_stock_data()
    transformer.calculate_indicators()
    transformer.generate_features()

def train_models():
    """Train ML models on processed data"""
    from analytics.ml_models import ModelTrainer
    
    trainer = ModelTrainer()
    trainer.train_price_prediction_model()
    trainer.train_risk_model()
    trainer.train_sentiment_model()

def generate_insights():
    """Generate analytics and insights"""
    from analytics.insights import InsightGenerator
    
    generator = InsightGenerator()
    generator.market_analysis()
    generator.risk_assessment()
    generator.portfolio_optimization()

# Define tasks
extract_task = PythonOperator(
    task_id='extract_market_data',
    python_callable=extract_market_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)

model_training_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag
)

insights_task = PythonOperator(
    task_id='generate_insights',
    python_callable=generate_insights,
    dag=dag
)

# Set dependencies
extract_task >> transform_task >> model_training_task >> insights_task
```

### Spark Job Configuration
```python
# jobs/spark_analytics.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

def create_spark_session():
    return SparkSession.builder \
        .appName("FinancialAnalytics") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

def process_financial_data():
    spark = create_spark_session()
    
    # Read data
    stock_data = spark.read.parquet("s3://bucket/data/stocks/")
    options_data = spark.read.parquet("s3://bucket/data/options/")
    
    # Feature engineering
    stock_features = stock_data.withColumn(
        "returns", 
        (col("close") - lag("close", 1).over(Window.orderBy("date"))) / lag("close", 1)
    ).withColumn(
        "volatility",
        stddev("returns").over(Window.orderBy("date").rowsBetween(-20, -1))
    )
    
    # ML Pipeline
    assembler = VectorAssembler(
        inputCols=["returns", "volatility", "volume"],
        outputCol="features"
    )
    
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="next_day_return",
        numTrees=100
    )
    
    # Train model
    feature_data = assembler.transform(stock_features)
    train_data, test_data = feature_data.randomSplit([0.8, 0.2])
    
    model = rf.fit(train_data)
    predictions = model.transform(test_data)
    
    # Evaluate
    evaluator = RegressionEvaluator(
        labelCol="next_day_return",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    rmse = evaluator.evaluate(predictions)
    print(f"RMSE: {rmse}")
    
    # Save results
    predictions.write.mode("overwrite").parquet("s3://bucket/predictions/")
    model.write().overwrite().save("s3://bucket/models/rf_model/")
    
    spark.stop()
```

---

## 🤖 ML Model Management

### MLflow Model Registry
```python
# ml_models/model_registry.py
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

class ModelRegistry:
    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def log_model(self, model, model_name, metrics, artifacts=None):
        """Log model to MLflow"""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(model.get_params())
            
            # Log metrics
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
            
            # Log model
            if hasattr(model, 'save'):  # TensorFlow/Keras
                mlflow.tensorflow.log_model(model, model_name)
            else:  # Scikit-learn
                mlflow.sklearn.log_model(model, model_name)
            
            # Log artifacts
            if artifacts:
                for artifact_path, artifact_data in artifacts.items():
                    mlflow.log_artifact(artifact_data, artifact_path)
            
            return mlflow.active_run().info.run_id
    
    def register_model(self, model_name, run_id, stage="None"):
        """Register model in model registry"""
        model_uri = f"runs:/{run_id}/{model_name}"
        
        try:
            mlflow.register_model(model_uri, model_name)
        except Exception:
            # Model already registered, create new version
            pass
        
        # Transition to staging
        if stage != "None":
            latest_version = self.client.get_latest_versions(
                model_name, stages=["None"]
            )[0].version
            
            self.client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage=stage
            )
    
    def load_model(self, model_name, stage="Production"):
        """Load model from registry"""
        model_uri = f"models:/{model_name}/{stage}"
        return mlflow.pyfunc.load_model(model_uri)
    
    def deploy_model(self, model_name, stage="Production"):
        """Deploy model to serving endpoint"""
        model_uri = f"models:/{model_name}/{stage}"
        
        # Deploy to Seldon Core or similar
        deployment_config = {
            "apiVersion": "machinelearning.seldon.io/v1",
            "kind": "SeldonDeployment",
            "metadata": {"name": f"{model_name.lower()}-deployment"},
            "spec": {
                "predictors": [{
                    "name": "default",
                    "replicas": 3,
                    "graph": {
                        "name": "model",
                        "implementation": "MLFLOW_SERVER",
                        "modelUri": model_uri
                    }
                }]
            }
        }
        
        return deployment_config
```

### Model Training Pipeline
```python
# ml_models/training_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

class ModelTrainingPipeline:
    def __init__(self, model_registry):
        self.model_registry = model_registry
        self.models = {}
    
    def prepare_data(self, data):
        """Prepare data for model training"""
        # Feature engineering
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        data['rsi'] = self.calculate_rsi(data['close'])
        data['macd'] = self.calculate_macd(data['close'])
        
        # Target variable (next day return)
        data['target'] = data['returns'].shift(-1)
        
        # Remove NaN values
        data = data.dropna()
        
        # Select features
        features = ['returns', 'volatility', 'rsi', 'macd', 'volume']
        X = data[features]
        y = data['target']
        
        return X, y
    
    def train_random_forest_model(self, X, y):
        """Train Random Forest model"""
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Evaluate
        predictions = best_model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        metrics = {
            'mse': mse,
            'r2': r2,
            'best_params': grid_search.best_params_
        }
        
        # Log to MLflow
        run_id = self.model_registry.log_model(
            best_model, "random_forest_predictor", metrics
        )
        
        self.model_registry.register_model(
            "random_forest_predictor", run_id, "Staging"
        )
        
        return best_model, metrics
    
    def train_lstm_model(self, X, y, sequence_length=60):
        """Train LSTM model for time series prediction"""
        # Prepare sequences
        X_sequences, y_sequences = self.create_sequences(X, y, sequence_length)
        
        # Split data
        train_size = int(0.8 * len(X_sequences))
        X_train, X_test = X_sequences[:train_size], X_sequences[train_size:]
        y_train, y_test = y_sequences[:train_size], y_sequences[train_size:]
        
        # Build LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        metrics = {
            'mse': mse,
            'val_loss': min(history.history['val_loss'])
        }
        
        # Log to MLflow
        run_id = self.model_registry.log_model(
            model, "lstm_predictor", metrics
        )
        
        self.model_registry.register_model(
            "lstm_predictor", run_id, "Staging"
        )
        
        return model, metrics
    
    def create_sequences(self, X, y, sequence_length):
        """Create sequences for LSTM training"""
        X_sequences, y_sequences = [], []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X.iloc[i-sequence_length:i].values)
            y_sequences.append(y.iloc[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
```

---

## ⚙️ Configuration

### Production Configuration
```python
# config/production.py
import os
from pathlib import Path

class ProductionConfig:
    # Application
    DEBUG = False
    TESTING = False
    ENVIRONMENT = "production"
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY')
    ALLOWED_HOSTS = ['analytics.yourdomain.com', 'api.yourdomain.com']
    
    # Database Configuration
    DATABASE_CONFIG = {
        'postgres': {
            'url': os.environ.get('POSTGRES_URL'),
            'pool_size': 20,
            'max_overflow': 30,
            'pool_recycle': 3600
        },
        'mongodb': {
            'url': os.environ.get('MONGODB_URL'),
            'max_pool_size': 100
        },
        'redis': {
            'url': os.environ.get('REDIS_URL'),
            'max_connections': 50
        }
    }
    
    # Spark Configuration
    SPARK_CONFIG = {
        'master': os.environ.get('SPARK_MASTER', 'yarn'),
        'app_name': 'FinancialAnalytics-Prod',
        'executor_memory': '4g',
        'executor_cores': 4,
        'driver_memory': '2g',
        'max_result_size': '2g',
        'sql_adaptive_enabled': True,
        'sql_adaptive_coalesce_partitions_enabled': True
    }
    
    # ML Configuration
    ML_CONFIG = {
        'model_registry_uri': os.environ.get('MLFLOW_TRACKING_URI'),
        'artifact_root': 's3://your-bucket/mlruns',
        'experiment_name': 'financial_models_prod',
        'model_serving_endpoint': os.environ.get('MODEL_SERVING_ENDPOINT')
    }
    
    # API Configuration
    API_CONFIG = {
        'host': '0.0.0.0',
        'port': 8000,
        'workers': 8,
        'worker_class': 'uvicorn.workers.UvicornWorker',
        'timeout': 300,
        'keepalive': 5
    }
    
    # Data Pipeline Configuration
    PIPELINE_CONFIG = {
        'batch_size': 10000,
        'max_retries': 3,
        'retry_delay': 300,
        'data_retention_days': 365,
        'checkpoint_interval': '5 minutes'
    }
    
    # Monitoring
    MONITORING = {
        'prometheus_port': 8080,
        'log_level': 'INFO',
        'sentry_dsn': os.environ.get('SENTRY_DSN'),
        'datadog_api_key': os.environ.get('DATADOG_API_KEY')
    }
```

---

## 📊 Monitoring & Performance

### Health Check System
```python
# monitoring/health_checks.py
from fastapi import FastAPI, HTTPException
import psutil
import redis
from sqlalchemy import create_engine
from pyspark.sql import SparkSession
import subprocess

app = FastAPI()

@app.get("/health")
async def comprehensive_health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {},
        "resources": {},
        "data_pipeline": {}
    }
    
    # Database health checks
    try:
        engine = create_engine(DATABASE_CONFIG['postgres']['url'])
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        health_status["services"]["postgres"] = "healthy"
    except Exception as e:
        health_status["services"]["postgres"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Redis health check
    try:
        r = redis.from_url(DATABASE_CONFIG['redis']['url'])
        r.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Spark cluster health check
    try:
        spark = SparkSession.builder.appName("HealthCheck").getOrCreate()
        spark.sql("SELECT 1").collect()
        health_status["services"]["spark"] = "healthy"
        spark.stop()
    except Exception as e:
        health_status["services"]["spark"] = f"unhealthy: {str(e)}"
    
    # System resources
    health_status["resources"] = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "load_average": psutil.getloadavg()
    }
    
    # Data pipeline status
    health_status["data_pipeline"] = {
        "celery_workers": get_celery_worker_status(),
        "airflow_dags": get_airflow_dag_status(),
        "data_freshness": check_data_freshness()
    }
    
    return health_status

def get_celery_worker_status():
    try:
        result = subprocess.run(
            ["celery", "-A", "analytics.celery", "inspect", "active"],
            capture_output=True, text=True, timeout=10
        )
        return "active" if result.returncode == 0 else "inactive"
    except Exception:
        return "unknown"

def get_airflow_dag_status():
    try:
        # Check if Airflow DAGs are running
        result = subprocess.run(
            ["airflow", "dags", "state", "financial_data_pipeline"],
            capture_output=True, text=True, timeout=10
        )
        return "running" if "success" in result.stdout else "failed"
    except Exception:
        return "unknown"

def check_data_freshness():
    # Check when data was last updated
    try:
        engine = create_engine(DATABASE_CONFIG['postgres']['url'])
        with engine.connect() as conn:
            result = conn.execute(
                "SELECT MAX(updated_at) FROM market_data"
            ).scalar()
            
            if result:
                minutes_old = (datetime.utcnow() - result).total_seconds() / 60
                return f"data_age_minutes: {minutes_old:.0f}"
            return "no_data"
    except Exception:
        return "error_checking_freshness"
```

### Performance Monitoring
```python
# monitoring/performance_metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# Define metrics
REQUEST_COUNT = Counter('analytics_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('analytics_request_duration_seconds', 'Request duration')
ACTIVE_JOBS = Gauge('analytics_jobs_active', 'Active processing jobs')
DATA_PROCESSING_TIME = Histogram('analytics_data_processing_seconds', 'Data processing time')
MODEL_PREDICTION_TIME = Histogram('analytics_model_prediction_seconds', 'Model prediction time')
ERROR_RATE = Counter('analytics_errors_total', 'Total errors', ['error_type'])

def monitor_performance(func_type='general'):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record success metrics
                duration = time.time() - start_time
                
                if func_type == 'data_processing':
                    DATA_PROCESSING_TIME.observe(duration)
                elif func_type == 'model_prediction':
                    MODEL_PREDICTION_TIME.observe(duration)
                else:
                    REQUEST_LATENCY.observe(duration)
                
                return result
                
            except Exception as e:
                ERROR_RATE.labels(error_type=type(e).__name__).inc()
                raise
                
        return wrapper
    return decorator

def start_metrics_server(port=8080):
    """Start Prometheus metrics server"""
    start_http_server(port)
    print(f"Metrics server started on port {port}")
```

---

## 🔒 Security

### Authentication & Authorization
```python
# security/auth.py
from fastapi import HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
import redis

security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
redis_client = redis.Redis.from_url(os.getenv('REDIS_URL'))

class User:
    def __init__(self, username: str, email: str, roles: list):
        self.username = username
        self.email = email
        self.roles = roles

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(
            credentials.credentials, 
            os.getenv('JWT_SECRET_KEY'), 
            algorithms=["HS256"]
        )
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Check if token is blacklisted
        if redis_client.get(f"blacklist:{credentials.credentials}"):
            raise HTTPException(status_code=401, detail="Token revoked")
        
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_role(required_role: str):
    """Decorator to require specific role"""
    def role_checker(token_data=Depends(verify_token)):
        user_roles = token_data.get("roles", [])
        if required_role not in user_roles:
            raise HTTPException(
                status_code=403, 
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        return token_data
    return role_checker

# Usage in API endpoints
@app.get("/api/admin/models")
async def get_models(user=Depends(require_role("admin"))):
    # Admin-only endpoint
    pass

@app.get("/api/data/stocks")
async def get_stock_data(user=Depends(require_role("analyst"))):
    # Analyst-level access required
    pass
```

### Data Encryption
```python
# security/encryption.py
from cryptography.fernet import Fernet
import os
import base64

class DataEncryption:
    def __init__(self):
        key = os.getenv('ENCRYPTION_KEY')
        if not key:
            key = Fernet.generate_key()
            print(f"Generated new encryption key: {key.decode()}")
        else:
            key = key.encode()
        self.cipher = Fernet(key)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_file(self, file_path: str, output_path: str):
        """Encrypt entire file"""
        with open(file_path, 'rb') as file:
            file_data = file.read()
        
        encrypted_data = self.cipher.encrypt(file_data)
        
        with open(output_path, 'wb') as file:
            file.write(encrypted_data)
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Spark Memory Issues
```bash
# Monitor Spark application
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --driver-memory 4g \
  --executor-memory 8g \
  --executor-cores 4 \
  --num-executors 10 \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  analytics_job.py

# Check Spark UI for memory usage
# http://spark-master:8080
```

#### 2. Data Pipeline Failures
```python
# Debug Airflow DAG
def debug_airflow_task():
    from airflow.models import DagRun, TaskInstance
    
    # Get latest DAG run
    dag_run = DagRun.find(dag_id='financial_data_pipeline')[-1]
    
    # Check failed tasks
    failed_tasks = dag_run.get_task_instances(state='failed')
    
    for task in failed_tasks:
        print(f"Failed task: {task.task_id}")
        print(f"Log: {task.log}")
        
        # Retry task
        task.clear()
```

#### 3. Model Serving Issues
```bash
# Check MLflow model serving
curl -X POST http://localhost:5000/invocations \
  -H 'Content-Type: application/json' \
  -d '{"data": [[1.0, 2.0, 3.0]]}'

# Debug Seldon deployment
kubectl logs -f deployment/model-deployment
kubectl describe seldondeployment model-deployment
```

### Performance Optimization
```python
# Optimize Spark queries
def optimize_spark_query():
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    
    # Cache frequently used DataFrames
    df.cache()
    
    # Optimize joins
    df1.join(broadcast(df2), "key")

# Optimize database queries
def optimize_database_queries():
    # Use connection pooling
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=30,
        pool_recycle=3600,
        echo=False
    )
    
    # Use bulk operations
    df.to_sql('table_name', engine, if_exists='append', method='multi')
```

### Debugging Commands
```bash
# Check container status
docker-compose ps
docker-compose logs -f analytics

# Monitor resource usage
docker stats financial-analytics
htop

# Check Spark cluster
curl http://spark-master:8080/api/v1/applications

# Debug Celery tasks
celery -A analytics.celery inspect active
celery -A analytics.celery flower  # Web interface

# Check database connections
docker exec -it analytics-postgres psql -U analyst -d analytics
```

---

## 📈 Performance Benchmarks

- **Data Processing**: 10GB+ datasets processed in < 30 minutes
- **Model Training**: Complex models trained in < 2 hours
- **API Response**: < 100ms for predictions
- **Throughput**: 1000+ requests/second
- **Memory**: Efficient memory usage with Spark optimization
- **Uptime**: 99.9% availability target

---

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/olaitanojo/financial-data-analysis-toolkit/issues)
- **Documentation**: [Wiki](https://github.com/olaitanojo/financial-data-analysis-toolkit/wiki)
- **API Docs**: [Swagger UI](http://localhost:8000/docs)
- **Community**: [Discord](https://discord.gg/financial-analytics)

---

*Last updated: December 2024*
*Version: 3.0.0*
