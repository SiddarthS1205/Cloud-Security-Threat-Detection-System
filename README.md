
# AI-Powered Cloud Security Threat Detection and Analytics System

An AI-powered security monitoring system that analyzes cloud security logs, identifies suspicious activities using machine learning, and presents security insights through an interactive Streamlit dashboard.

## 📌 Project Information

- **Project Title:** AI-Powered Cloud Security Threat Detection and Analytics System
- **Student Name:** SIDDARTH S
- **Register Number:** 23UAI098
- **Project Guide:** V. SELVI
- **Domain:** Artificial Intelligence, Cloud Computing, and Cybersecurity

## 🎯 Project Overview

Cloud environments generate large volumes of security logs that are difficult to monitor manually. This project combines Artificial Intelligence, Data Engineering, Cloud log processing, and Security Engineering to help identify potentially suspicious activities.

The system processes security events, extracts relevant features, classifies events using a trained Random Forest model, stores data in an SQLite database, and displays results through an interactive Streamlit dashboard.

## 🎯 Objectives

- Analyze cloud-style security logs automatically.
- Detect potentially suspicious activities using machine learning.
- Develop a data pipeline for log generation, validation, cleaning, and storage.
- Process CloudTrail-format JSON events through an offline ingestion pipeline.
- Visualize security events through an interactive dashboard.
- Implement input validation, anonymisation, and security controls.
- Demonstrate the integration of all four CPPE modules.

## 🧩 Four CPPE Modules

### 1. AI Engineering

- Random Forest classification model.
- Feature engineering and preprocessing.
- Model training and evaluation.
- Classification of normal and suspicious security events.
- Saved model and feature-engineering artifacts.

### 2. Data Engineering

- Security-log data generation.
- Data validation and cleaning.
- Feature preparation for machine learning.
- SQLite database storage.
- Automated data-processing pipeline.

### 3. Cloud Engineering

- CloudTrail-format JSON event parsing.
- Mapping cloud event fields to the project's security-log format.
- Offline event ingestion and reporting.
- Processing of sample CloudTrail-format events.

**Current limitation:** Live AWS CloudTrail integration has not been implemented. The current ingestion pipeline processes sample JSON data offline.

### 4. Security Engineering

- Suspicious-event identification.
- CSV input validation.
- Log anonymisation.
- IP allowlisting functionality.
- Incident-management dashboard.
- Security-log analysis and monitoring.

These features demonstrate core security controls. The application is intended for academic demonstration and is not yet production-ready.

## 🏗️ System Architecture

```text
        Synthetic Security Logs
                  |
                  v
       Data Validation and Cleaning
                  |
                  v
           Feature Engineering
                  |
                  v
           SQLite Database
                  |
                  v
        Random Forest Classifier
                  |
                  v
         Threat Detection Results
                  |
                  v
       Streamlit Security Dashboard
```

The offline CloudTrail-format ingestion pipeline separately processes sample JSON events and maps them into the project's security-log format.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11 |
| Machine Learning | Scikit-learn, Random Forest |
| Data Processing | Pandas, NumPy |
| Database | SQLite |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Cloud Log Processing | CloudTrail-format JSON |
| Testing | Pytest |
| Version Control | Git and GitHub |

## 📂 Project Structure

```text
Cloud-Security-Threat-Detection-System/
│
├── dashboard/
│   ├── app.py
│   ├── db_utils.py
│   └── csv_validator.py
│
├── src/
│   ├── data_pipeline/
│   │   ├── generator.py
│   │   ├── validator.py
│   │   ├── cleaner.py
│   │   ├── features.py
│   │   └── storage.py
│   │
│   └── cloud/
│       ├── parser.py
│       ├── anonymiser.py
│       ├── mapper.py
│       ├── ingestor.py
│       ├── reporter.py
│       └── ip_allowlist.py
│
├── models/
│   ├── threat_classifier.pkl
│   ├── feature_engineer.pkl
│   └── training_metadata.json
│
├── tests/
│   └── sample_cloudtrail.json
│
├── docs/
│   └── screenshots/
│       ├── security-overview.png
│       ├── log-detection.png
│       ├── model-performance.png
│       ├── incident-management.png
│       └── cppe-module-overview.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

*This is the intended overview of the principal project files. Your actual repository may omit generated data, databases, or model artifacts through `.gitignore`.*

## 🖥️ Application Output Screenshots

### 1. Security Overview Dashboard

Displays security event summaries, normal and suspicious event counts, and available visualizations.

![Security Overview Dashboard](docs/screenshots/security-overview.png)

### 2. Log Upload and Threat Detection

Displays the log upload interface and the results of security-event analysis.

![Log Upload and Detection](docs/screenshots/log-detection.png)

### 3. Model Performance

Displays the machine-learning model's evaluation metrics.

![Model Performance Dashboard](docs/screenshots/model-performance.png)

### 4. Incident Management

Displays the available incident-management interface and security incident records.

![Incident Management Dashboard](docs/screenshots/incident-management.png)

### 5. CPPE Module Overview

Displays the application's overview of the four CPPE modules.

![CPPE Module Overview](docs/screenshots/cppe-module-overview.png)

> **Note:** Replace these image paths with the exact filenames of your actual screenshots. GitHub will display the screenshots only when the referenced files exist in the repository.

## 📊 Model Evaluation

The Random Forest classifier was evaluated using a held-out portion of the synthetic dataset.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 98.20% |
| Macro Precision | 96.56% |
| Macro Recall | 97.94% |
| Suspicious-Class F1-Score | 95.59% |
| False Positives | 13 |
| False Negatives | 5 |

**Important:** These results were obtained using synthetic data. They do not establish equivalent performance on real-world cloud security logs. False-positive and false-negative counts refer to the reported evaluation set.

## 📈 Dashboard Features

The Streamlit dashboard includes the following pages:

### Security Overview
Provides a summary of available security events and detection counts.
<img width="1570" height="910" alt="image" src="https://github.com/user-attachments/assets/6ac329ca-87bf-47de-b74a-c58df9f22859" />


### Log Upload & Detection
Allows users to upload supported log files for validation and analysis.
<img width="1572" height="913" alt="image" src="https://github.com/user-attachments/assets/aec2c973-db47-4b95-93c9-8d0c21df882d" />
<img width="1570" height="912" alt="image" src="https://github.com/user-attachments/assets/1ada9b3d-7a44-44c9-8388-6aa21c751e71" />


### Model Performance
Presents the model evaluation information.
<img width="1560" height="838" alt="image" src="https://github.com/user-attachments/assets/d0f6b23a-34fe-4bc7-88c3-2156e96707b8" />


### Incident Management
Provides an interface to review and manage displayed security incidents.
<img width="1564" height="674" alt="image" src="https://github.com/user-attachments/assets/2f3d7dc1-903f-460a-87a2-0285e5ef3064" />


### CPPE Module Overview
Describes the project's coverage of AI Engineering, Data Engineering, Cloud Engineering, and Security Engineering.
<img width="1430" height="865" alt="image" src="https://github.com/user-attachments/assets/db34bf07-ebc6-49cf-b186-b229bbfe5c49" />


The dashboard combines synthetic security records with unlabelled CloudTrail-format sample events. Unlabelled events must not automatically be considered confirmed normal activity.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.11
- Git
- pip

### Step 1: Clone the Repository

```powershell
git clone https://github.com/SiddarthS1205/Cloud-Security-Threat-Detection-System.git
cd Cloud-Security-Threat-Detection-System
```

### Step 2: Create a Virtual Environment

```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the virtual environment's Python executable directly.

### Step 4: Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run the Dashboard

```powershell
python -m streamlit run dashboard/app.py
```

Open the local URL printed in the terminal. Streamlit commonly uses:

```text
http://localhost:8501
```

### Step 6: Run Tests

```powershell
python -m pytest
```

Run the command from the project root directory. Test results depend on the installed dependencies and the current project files.

## 🔐 Security Considerations

- Never commit AWS access keys, secret keys, passwords, or `.env` files.
- Do not publish confidential or personally identifiable cloud logs.
- Use synthetic or appropriately anonymised data for demonstrations.
- Validate uploaded files before processing them.
- Protect sensitive database files and model artifacts.
- Review authentication, access control, file-upload handling, database protection, and model-integrity controls before deployment.

## 🚧 Current Limitations

- Live AWS CloudTrail integration is not available.
- CloudTrail-format events are processed offline.
- Model evaluation is based on synthetic data.
- The model has not been validated against a representative real-world cloud security dataset.
- Additional security hardening is required before production deployment.

## 🔮 Future Enhancements

- Integrate live AWS CloudTrail events through securely configured AWS services.
- Evaluate the model using real, appropriately anonymised security datasets.
- Implement secure authentication and role-based access control.
- Add alert notifications for high-risk events.
- Improve incident investigation and audit trails.
- Deploy the application to a secure cloud environment.
- Add model-drift monitoring and periodic retraining.

## 👨‍💻 Author

**SIDDARTH S**

B.Tech — Artificial Intelligence and Data Science

Project: AI-Powered Cloud Security Threat Detection and Analytics System

## 📜 Disclaimer

This project is developed for academic learning and demonstration. Detection results are indicative and should not be used as the sole basis for real-world security decisions.
