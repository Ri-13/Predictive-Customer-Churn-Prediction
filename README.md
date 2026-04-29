🚀 Project Overview

The system is built to identify customers at risk of leaving a service provider by analyzing usage patterns, contract details, and demographic data. By integrating Spark MLlib for training and Spark Structured Streaming for deployment, the project enables businesses to move from reactive to proactive retention strategies.

🏗️ System Architecture

Phase 1: Batch Training & Optimization

The first phase focuses on building a high-performance predictive engine:

Data Preprocessing: Handles missing values (e.g., TotalCharges imputation) and performs automated class weight balancing to address data imbalance.

Feature Engineering: Implements a robust pipeline using StringIndexer, OneHotEncoder, and VectorAssembler to prepare categorical and numeric data.

Model Comparison: Trains and evaluates multiple models, including Naive Bayes, Decision Trees, and the proposed Scalable Random Forest (SRF).

Hyperparameter Tuning: Utilizes 5-fold Cross-Validation with a parameter grid to optimize the SRF model for maximum AUC-ROC and F1-score.

Phase 2: Near Real-Time Streaming

The second phase deploys the best-performing model into a live production-like environment:

Live Monitoring: Watches a designated directory for incoming CSV data files.

Automated Prediction: Processes new data in 10-second micro-batches using the pre-trained SRF pipeline.

Actionable Enrichment: Categorizes customers into Risk Tiers (High, Medium, or Low Risk) based on their calculated churn probability.

Dual-Sink Output: Simultaneously streams results to the system console and writes them to persistent CSV storage for business reporting.

🛠️ Technical Stack

Language: Scala

Framework: Apache Spark (SQL, MLlib, Structured Streaming)

ML Algorithm: Scalable Random Forest (SRF)

Data Source: Telco Customer Churn Dataset

📊 Sample Output

Upon execution, the system provides:

Model Metrics: AUC-ROC, Accuracy, Precision, Recall, and F1 Score.

Feature Importance: A ranked list of the top 10 predictors of churn.

Real-Time Alerts: A live dashboard view of customers likely to churn as data arrives.
