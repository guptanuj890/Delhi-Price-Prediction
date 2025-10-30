# Delhi Price Prediction 🏡

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Deployment](https://img.shields.io/badge/Deployment-Flask%20%7C%20Docker-brightgreen?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://github.com/guptanuj890/Delhi-Price-Prediction/blob/main/LICENSE)

***

## 1. Project Overview

The **Delhi Price Prediction** project is an end-to-end Machine Learning pipeline designed to predict prices for commodities or real estate within the Delhi National Capital Region (NCR).

This repository encompasses the entire ML lifecycle, from **Exploratory Data Analysis (EDA)** and **Model Training** to **Continuous Integration/Continuous Deployment (CI/CD)** ready code, complete with a web interface for making real-time predictions. The project utilizes industry-standard best practices, including modular code structure, configuration management, and model tracking via tools like MLflow (inferred from `mlruns`).

***

## 2. Key Features and Goal

* **Objective:** To accurately estimate the target price (e.g., house price, car price, etc.) in Delhi based on relevant features.
* **Modular Coding:** Code is structured into dedicated modules (`src`) for data ingestion, transformation, and model training, ensuring high maintainability and testability.
* **Web Application:** A user-friendly web interface (`app.py`, `templates`, `static`) built using **Flask** for interactive predictions.
* **Model Tracking:** Integration with a system (e.g., MLflow, inferred from `mlruns`) to track experiments, parameters, metrics, and artifact storage (`artifacts`).
* **Containerization:** Ready-to-deploy application using **Docker** (`Dockerfile`).

***

## 3. Technologies and Tools

This project is built using the following major technologies:

| Category | Tool/Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python (3.x) | Core programming language for ML and backend. |
| **Data & ML** | Pandas, NumPy, Scikit-learn | Data manipulation, processing, and model implementation. |
| **Web Framework** | Flask | Serving the trained model via a lightweight web API and UI. |
| **Deployment** | Docker | Containerizing the application for consistent deployment across environments. |
| **MLOps** | MLflow (Inferred) | Experiment tracking and model management. |
| **Notebooks** | Jupyter Notebook | Initial data exploration and model prototyping. |

***

## 4. Setup and Installation

Follow these steps to set up the project locally.

### Prerequisites

* Python 3.8+
* Git
* Docker (Optional, for containerized deployment)

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/guptanuj890/Delhi-Price-Prediction.git](https://github.com/guptanuj890/Delhi-Price-Prediction.git)
    cd Delhi-Price-Prediction
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Linux/Mac
    .\venv\Scripts\activate    # On Windows
    ```

3.  **Install Dependencies:**
    All necessary packages are listed in `requirements.txt` and can be installed using the `setup.py` file.
    ```bash
    pip install -e .
    ```

***

## 5. How to Run

### A. Run Locally (Development)

1.  Ensure you have completed the **Setup and Installation** steps.
2.  Run the main application file:
    ```bash
    python app.py
    ```
3.  The application will start running on `http://127.0.0.1:5000/`. Open this URL in your browser to access the prediction interface.

### B. Run with Docker (Production Ready)

1.  Ensure you have **Docker** installed and running on your system.
2.  Build the Docker image:
    ```bash
    docker build -t delhi-price-predictor .
    ```
3.  Run the container:
    ```bash
    docker run -p 5000:5000 delhi-price-predictor
    ```
4.  Access the application at `http://0.0.0.0:5000/`.

***

## 6. Project Structure

The repository follows a clean, structured layout for easy navigation and maintenance: