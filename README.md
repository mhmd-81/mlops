<p align="center">
  <img
    width="200"
    height="200"
    src="https://i.imgur.com/LcOx8t1.png"
    alt="Project logo"
  >
</p>
<h1 align="center">MLOps Pipeline</h1>

<p align="center">
  A hands-on project for learning and practicing MLOps concepts.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-active-success.svg" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Docker-Compose-blue.svg" alt="Docker">
  <img src="https://img.shields.io/badge/Airflow-3.3.0-orange.svg" alt="Airflow">
  <img src="https://img.shields.io/badge/MLflow-tracking-blue.svg" alt="MLflow">
</p>

---

## 📝 Table of Contents

- [About](#-about)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the Pipeline](#-running-the-pipeline)
- [Running the Tests](#-running-the-tests)
- [Usage](#-usage)
- [Built Using](#-built-using)
- [TODO](#-todo)
- [Authors](#-authors)

---

## 🧐 About

This project was created as a hands-on project for learning and improving my MLOps skills.

The main goal is to understand how machine learning models can be trained, tracked, tested, and automated using common MLOps tools.

The project focuses on building a simple pipeline that combines:

- Model training
- Experiment tracking
- Workflow orchestration
- Docker containerization
- Automated testing
- Model and training logs

This project is mainly intended for learning and experimentation, with the goal of gradually developing it into a more complete MLOps system.

---

## 📁 Project Structure

```text
mlops
├── airflow
├── data
├── Dockerfile
├── dockerfile.tests
├── mlartifacts
├── mlflow.db
├── notebooks
├── README.md
├── requirements.txt
├── src
├── tests
└── venv
```

---

## Getting Started

**install dependencies**
```
python3 -m venv venv
```
```
source venv/bin/activate
```
```
pip install -r requirements.txt
```