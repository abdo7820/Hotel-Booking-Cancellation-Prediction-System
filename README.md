# 🏨 Hotel Booking Cancellation Prediction --- Production ML System

> **End-to-end MLOps system for predicting hotel booking cancellations
> using multiple Machine Learning models, reproducible artifacts,
> FastAPI, Docker, an interactive dashboard, and production
> deployment.**

------------------------------------------------------------------------

## 📌 Project Overview

This project transforms a Hotel Booking Cancellation Machine Learning
workflow into a **production-ready application**.

The workflow is split into two main environments:

-   **Training Environment --- Kaggle:** EDA, preprocessing, feature
    engineering, model training, hyperparameter tuning, and ensemble
    models.
-   **Production Environment:** FastAPI backend, interactive dashboard,
    Dockerized deployment, API testing with Postman, and hosting.

The trained models and preprocessing objects are exported as reusable
artifacts, so the production server can make predictions **without
requiring GPU resources**.

### 🎯 Main Goal

Given the raw information of a hotel reservation, the system predicts:

-   Whether the booking is likely to be **Canceled** or **Not Canceled**
-   The **cancellation probability**, when the selected model supports
    `predict_proba`

------------------------------------------------------------------------

# 🏗️ System Architecture

``` text
                    ┌──────────────────────────────┐
                    │        HOTEL BOOKINGS        │
                    │       hotel_bookings.csv     │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      KAGGLE / TRAINING       │
                    │                              │
                    │  • EDA                       │
                    │  • Preprocessing             │
                    │  • Feature Engineering       │
                    │  • Model Training            │
                    │  • Hyperparameter Tuning     │
                    │  • Ensembles                 │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │          ARTIFACTS            │
                    │                              │
                    │  models/*.pkl                │
                    │  preprocessing.pkl           │
                    │  metrics.json                │
                    │  insights.json               │
                    │  feature_importance.json     │
                    │  feature_schema.json         │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │           GitHub             │
                    │      Version Control         │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       Docker Container       │
                    │                              │
                    │       FastAPI Backend        │
                    │              │               │
                    │              ▼               │
                    │       Interactive Dashboard  │
                    └──────────────┬───────────────┘
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                 ┌──────────────┐     ┌──────────────┐
                 │   Postman    │     │   Production │
                 │ API Testing  │     │    Hosting   │
                 └──────────────┘     │   Railway    │
                                      └──────────────┘
```

------------------------------------------------------------------------

# ✨ Key Features

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  🤖 Multiple Models                  Train and compare several ML
                                      algorithms

  🎯 Hyperparameter Tuning            Tuned XGBoost models for F1 /
                                      Accuracy

  🔗 Ensembles                        Soft Voting and Stacking models

  📦 Reproducible Artifacts           Models, preprocessing, metrics, and
                                      insights are exported

  ⚡ FastAPI                          REST API for health checks, model
                                      information, analytics, and
                                      prediction

  📊 Interactive Dashboard            Model comparison, feature
                                      importance, EDA insights, and
                                      prediction form

  🐳 Docker                           Consistent production environment

  🧪 Postman                          API endpoint validation

  🚀 Railway Deployment               Production hosting with HTTPS and
                                      automatic deployment from GitHub

  🖥️ No GPU in Production             Training can use Kaggle resources;
                                      inference runs on the server
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🧠 Machine Learning Pipeline

The training pipeline includes:

### 1. Exploratory Data Analysis

The project generates analytical insights such as:

-   Overall cancellation rate
-   Cancellation rate by season
-   Cancellation rate by deposit type
-   Cancellation rate by market segment
-   Cancellation rate by lead-time bucket
-   Chi-Square statistical analysis

### 2. Preprocessing

The preprocessing pipeline stores the transformations required by the
production API, including:

-   Frequency mappings
-   Power transformation
-   Ordinal encoding
-   Meal encoding
-   Scaling
-   One-hot encoding
-   Final feature-column alignment

### 3. Feature Engineering

The production prediction flow recreates the required feature
engineering for a single booking, including features such as:

-   Total nights
-   Total guests
-   Family indicator
-   Room type change indicator
-   Previous stays
-   Returning-customer indicator
-   Arrival season
-   ADR per person
-   Estimated total cost
-   Lead-time bucket
-   Previous cancellation indicator

### 4. Model Training

The training pipeline supports multiple algorithms, including models
such as:

-   Logistic Regression
-   Decision Tree
-   Random Forest
-   Gradient Boosting
-   AdaBoost
-   KNN
-   Naive Bayes
-   XGBoost
-   Tuned XGBoost
-   Soft Voting
-   Stacking

The available models are compared using evaluation metrics and exposed
through the dashboard/API.

------------------------------------------------------------------------

# 📦 Artifacts

After training, the project produces an `artifacts/` directory:

``` text
artifacts/
├── models/
│   ├── *.pkl
│   └── ...
├── preprocessing.pkl
├── metrics.json
├── insights.json
├── feature_importance.json
└── feature_schema.json
```

### Artifact responsibilities

#### `models/*.pkl`

Serialized trained ML models used by the production API.

#### `preprocessing.pkl`

Contains the preprocessing objects and mappings required to transform
incoming booking data exactly as expected by the trained models.

#### `metrics.json`

Contains model evaluation results and is used by the API/dashboard to
display model comparisons.

#### `insights.json`

Contains EDA statistics consumed by the dashboard.

#### `feature_importance.json`

Contains the most important model features displayed in the dashboard.

#### `feature_schema.json`

Describes the raw input structure expected by the prediction endpoint.

------------------------------------------------------------------------

# ⚙️ Project Structure

``` text
hotel_cancellation_mlops/
│
├── app/
│   └── main.py
│
├── static/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── config.js
│
├── artifacts/
│   ├── models/
│   ├── preprocessing.pkl
│   ├── metrics.json
│   ├── insights.json
│   ├── feature_importance.json
│   └── feature_schema.json
│
├── train.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── postman_collection.json
└── README.md
```

------------------------------------------------------------------------

# 🚀 Getting Started

## 1. Clone the Repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd hotel_cancellation_mlops
```

------------------------------------------------------------------------

## 2. Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🧪 Training on Kaggle

Training is designed to run once using the training dataset.

Example:

``` bash
python train.py \
  --data /kaggle/input/<dataset>/hotel_bookings.csv \
  --out artifacts
```

After training:

1.  Verify that `artifacts/` was generated.
2.  Download the complete `artifacts/` directory.
3.  Place it in the project root beside the `Dockerfile`.

> ⚠️ The versions of `scikit-learn`, `xgboost`, and `imbalanced-learn`
> used to load the serialized models should match, or be close to, the
> versions used during training.

------------------------------------------------------------------------

# 💻 Run Locally

Start the FastAPI application:

``` bash
uvicorn app.main:app --reload
```

### API

``` text
http://localhost:8000/docs
```

### Dashboard

``` text
http://localhost:8000/dashboard
```

The FastAPI application serves the dashboard from the `static/`
directory.

------------------------------------------------------------------------

# 🐳 Run with Docker

Build and start the application:

``` bash
docker compose up --build
```

The `artifacts/` directory is mounted as a read-only volume, allowing
the trained models and JSON artifacts to be updated without rebuilding
the application image.

------------------------------------------------------------------------

# 🔌 API Endpoints

  Method   Endpoint      Description
  -------- ------------- -----------------------------------
  `GET`    `/health`     API and loaded-model health check
  `GET`    `/models`     Available models sorted by F1
  `GET`    `/features`   Feature importance
  `GET`    `/insights`   EDA insights and statistics
  `GET`    `/schema`     Raw prediction input schema
  `POST`   `/predict`    Make a cancellation prediction

------------------------------------------------------------------------

# 🔮 Prediction API

## Request

``` json
{
  "model_key": "xgboost",
  "booking": {
    "hotel": "Resort Hotel",
    "lead_time": 30,
    "arrival_date_month": "July",
    "arrival_date_year": 2017,
    "arrival_date_day_of_month": 15,
    "arrival_date_week_number": 24,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "agent": 9,
    "company": null,
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 100,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0
  }
}
```

## Response

``` json
{
  "model_used": "xgboost",
  "prediction": 0,
  "prediction_label": "Not Canceled",
  "cancellation_probability": 0.1234
}
```

------------------------------------------------------------------------

# 📊 Dashboard

The dashboard provides five main areas:

### Overview

High-level booking and cancellation statistics plus visual analytics.

### Models

Comparison of the available ML models using:

-   F1
-   Accuracy
-   Precision
-   Recall
-   ROC-AUC

### Feature Importance

Displays the most influential features used by the selected model.

### Insights

Visualizes:

-   Cancellation by season
-   Cancellation by deposit type
-   Cancellation by market segment
-   Cancellation by lead-time bucket
-   Chi-Square statistical results

### Prediction Studio

Users can:

1.  Select a trained model.
2.  Enter booking information.
3.  Send the booking to the FastAPI backend.
4.  Receive the prediction and cancellation probability.

------------------------------------------------------------------------

# 🧪 API Testing with Postman

Import:

``` text
postman_collection.json
```

Recommended testing order:

``` text
1. Health Check
2. List Models
3. Feature Importance
4. Insights
5. Raw Feature Schema
6. Predict
7. Predict - Invalid Model
```

For the prediction request, use the model `file` value returned from
`/models` without the `.pkl` extension.

Example:

``` text
xgboost
```

The invalid-model request should return:

``` text
HTTP 404
```

------------------------------------------------------------------------

# 🚀 Production Deployment

The application is designed to be deployed as a Dockerized FastAPI
service.

## Railway

The production deployment uses **Railway**.

Typical deployment flow:

``` text
GitHub Repository
       │
       ▼
Railway
       │
       ▼
Docker Build
       │
       ▼
FastAPI Application
       │
       ├── /docs
       └── /dashboard
```

Railway provides:

-   GitHub-based deployment
-   Docker build/deployment
-   HTTPS
-   Automatic `PORT` environment variable
-   Public production URL
-   Scalable hosting

After deployment, the application is available through:

``` text
https://<your-app>.up.railway.app
```

Dashboard:

``` text
https://<your-app>.up.railway.app/dashboard
```

API documentation:

``` text
https://<your-app>.up.railway.app/docs
```

------------------------------------------------------------------------

# 🔐 Environment Variables

The backend supports the following configuration variables:

  Variable            Purpose
  ------------------- ----------------------------------------------
  `PORT`              Server port supplied by the hosting platform
  `ARTIFACTS_DIR`     Location of the ML artifacts
  `STATIC_DIR`        Location of dashboard static files
  `ALLOWED_ORIGINS`   Allowed CORS origins

Example:

``` text
ARTIFACTS_DIR=artifacts
STATIC_DIR=static
ALLOWED_ORIGINS=*
```

For production, CORS can be restricted to the actual frontend domain.

------------------------------------------------------------------------

# 🔄 Production Prediction Flow

``` text
User
  │
  ▼
Dashboard
  │
  │ POST /predict
  ▼
FastAPI
  │
  ▼
Feature Engineering
  │
  ▼
Saved Preprocessing Pipeline
  │
  ▼
Selected ML Model
  │
  ▼
Prediction
  │
  ├── Canceled / Not Canceled
  └── Cancellation Probability
  │
  ▼
Dashboard Result
```

The production server loads the artifacts during application startup and
reuses them for prediction requests.

------------------------------------------------------------------------

# 🛡️ Production Considerations

### Model compatibility

Serialized `.pkl` files depend on compatible library versions.

Keep training and production versions aligned, especially for:

``` text
scikit-learn
xgboost
imbalanced-learn
```

### Large models

Some ensemble models can be relatively large. If repository or image
size becomes a concern, weaker or unnecessarily large models can be
removed from `artifacts/models/` while keeping the best-performing
models.

The backend dynamically loads available `.pkl` files and filters the
displayed metrics to models that actually exist.

------------------------------------------------------------------------

# 📈 Extending the Project

Adding a new trained model can be done by:

1.  Saving the model as a `.pkl` file inside:

``` text
artifacts/models/
```

2.  Adding its evaluation result to:

``` text
artifacts/metrics.json
```

The API and dashboard can then discover the model through the existing
model-loading mechanism.

------------------------------------------------------------------------

# 🧩 Tech Stack

### Machine Learning

-   Python
-   Pandas
-   Scikit-learn
-   XGBoost
-   Imbalanced-learn
-   Joblib

### Backend

-   FastAPI
-   Uvicorn
-   Pydantic

### Frontend

-   HTML
-   CSS
-   JavaScript
-   Chart.js

### DevOps / MLOps

-   Git
-   GitHub
-   Docker
-   Docker Compose
-   Railway
-   Postman

### Training Infrastructure

-   Kaggle

------------------------------------------------------------------------

# 📋 Production Checklist

Before deployment:

``` text
☐ Train models on Kaggle
☐ Generate artifacts/
☐ Verify preprocessing.pkl
☐ Verify models/*.pkl
☐ Verify metrics.json
☐ Verify insights.json
☐ Verify feature_importance.json
☐ Verify feature_schema.json
☐ Test FastAPI locally
☐ Test /health
☐ Test /models
☐ Test /features
☐ Test /insights
☐ Test /schema
☐ Test /predict
☐ Test invalid model handling
☐ Test dashboard
☐ Build Docker image
☐ Deploy to Railway
☐ Test production /health
☐ Test production /docs
☐ Test production /dashboard
```

------------------------------------------------------------------------

# 🎯 Project Outcome

This project converts a traditional notebook-based Machine Learning
workflow into a complete production system:

``` text
EDA
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
Multiple ML Models
 ↓
Tuning & Ensembles
 ↓
Model Evaluation
 ↓
Reproducible Artifacts
 ↓
FastAPI REST API
 ↓
Interactive Dashboard
 ↓
Docker
 ↓
GitHub
 ↓
Railway
 ↓
Production Prediction
```

**The result is a complete ML application that separates model training
from production inference, exposes the trained models through an API,
visualizes the model and business insights, and provides an interactive
prediction interface.**
