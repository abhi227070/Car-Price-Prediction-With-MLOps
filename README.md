# Car Price Prediction using ML Ops

This project is focused on implementing **ML Ops** for car price prediction. The main goal is to build a pipeline that automates the process of model training, evaluation, and prediction, while integrating essential ML Ops practices like logging, error handling, and version control for models.

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Tools Used](#tools-used)
- [Getting Started](#getting-started)
- [Screenshots](#screenshots)
- [Use Case](#use-case)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
- [License](#license)

---

## Introduction

This repository contains a machine learning project aimed at predicting car prices using a combination of ML models and ML Ops practices. The project is structured to focus on continuous integration and deployment (CI/CD) for ML models, making it an example of good ML Ops practices.

## Project Overview

The goal of this project is to predict car prices based on various features such as car make, model, year, mileage, etc. It also focuses on the use of ML Ops tools and pipelines to manage the model lifecycle, including model training, monitoring, and deployment.

### Key Components:
- **ML Pipeline**: A series of stages to load data, preprocess it, train models, and make predictions.
- **Model Management**: Proper management of trained models using `joblib` for saving and loading models.
- **Logging and Exception Handling**: Robust logging and error handling during data ingestion, transformation, and training.

---

## File Structure

The file structure of the project is organized as follows:

Car-Price-Prediction-Using-ML-Ops/
│
├── artifacts/                    # Directory for storing raw, train, test datasets and trained models
│   ├── raw.csv                   # Raw dataset
│   ├── train.csv                 # Training dataset
│   ├── test.csv                  # Testing dataset
│   ├── preprocessor.pkl          # Preprocessing model (e.g., scaling, encoding)
│   └── model.pkl                 # Trained model
│
├── logs/                         # Folder for logging pipeline and model monitoring information
│
├── src/                          # Source code directory for the project
│   ├── car_price_prediction/     # Main module for car price prediction
│   │   ├── components/           # Core components of the pipeline
│   │   │   ├── __init__.py       # Package initialization
│   │   │   ├── data_ingestion.py # Code for ingesting raw data
│   │   │   ├── data_transformation.py # Code for transforming data
│   │   │   ├── model_training.py # Code for training the model
│   │   │   └── model_monitoring.py # Code for monitoring model performance
│   │   ├── pipelines/            # MLOps pipelines for training and prediction
│   │   │   ├── __init__.py       # Package initialization
│   │   │   ├── training_pipeline.py # Defines the pipeline for model training
│   │   │   └── prediction_pipeline.py # Defines the pipeline for model inference
│   │   ├── notebooks/            # Jupyter Notebooks for EDA and feature engineering
│   │   │   ├── data/             # Raw data for notebooks
│   │   │   ├── EDA.ipynb         # Exploratory Data Analysis (EDA)
│   │   │   └── feature_engineering.ipynb # Feature engineering process
│   │   ├── __init__.py           # Package initialization
│   │   ├── exception.py          # Custom exception handling
│   │   ├── logger.py             # Logging configuration
│   │   └── utils.py              # Utility functions (e.g., model saving, evaluation metrics)
│
├── .gitignore                    # Git ignore file to exclude unnecessary files from version control
├── Dockerfile                    # Docker configuration for containerizing the project
├── LICENSE                       # Project license information
├── README.md                     # Project documentation (this file)
├── app.py                        # Flask/Django app for model serving and prediction API
├── setup.py                      # Installation script for setting up the environment
├── template.py                   # Template script for adding new components/pipelines
└── requirements.txt              # List of project dependencies
