# Deep-learning-Cable-Failure-Predection-
A Model that predicts failure of cables

# Cable Failure Risk Prediction Using Deep Learning (15-KV XLPE Underground Cable Dataset)

## 📌 Project Idea
We built a **Deep Learning system (MLP Neural Network)** that predicts the **Risk Level** of underground power cables (**Low – Medium – High**) using operational features such as:
*   **Age**
*   **Partial Discharge**
*   **Visual Condition**
*   **Corrosion**
*   **Electrical Loading**

The goal is to help electrical distribution companies **detect cables that may fail soon** → reduce outages → optimize maintenance → avoid emergencies.

---

## 📌 The Problem We Solve
Electricity companies struggle with:
*   **Sudden underground cable failures**
*   **No automated tool** to assess cable health
*   **Manual inspections** are slow & inaccurate

Our model predicts failure risk based on real operational signals to help engineers make preventive decisions.

---

## 📌 Data Source
**Dataset used:** "15-KV XLPE Underground Cable Dataset"

It contains thousands of cable records with:
*   Operational age
*   Partial discharge readings
*   Visual inspection codes
*   Corrosion level
*   Load currents

We cleaned, analyzed, and built a **Health Index + Health Class** that the model learns to predict.

---

## 📌 Tech Stack

| Category | Technology |
|----------|------------|
| **Deep Learning** | PyTorch |
| **API Backend** | FastAPI |
| **Frontend** | React + Vite |
| **Charts** | Recharts / Chart.js |
| **Data Files** | Pandas + NumPy |
| **Model Serving** | TorchScript / .pt |
| **Deployment** | .bat files |

---

## 👤 Team Structure & Roles

### 1. Tayam — Data Engineer (Data Pipeline Owner)
**Responsibilities:**
*   Load and clean dataset
*   Handle missing values & outliers
*   Exploratory Data Analysis & Visualizations
*   Create Health Index and Health Class (Labels)
*   Train/Test split using Stratification
*   Save processed datasets as PKL
*   Organize project files and ensure reproducibility

**Files:**
*   `Cable_Failure_Prediction.ipynb`
*   `inspect_data.py`
*   `inspect_columns.py`
*   `check_columns.py`
*   `regenerate_data.py`

### 2. Abdullah — Feature Engineering + Model Designer
**Responsibilities:**
*   Select best predictive features
*   Apply StandardScaler
*   One-Hot Encoding
*   Design MLP Deep Learning Model
*   Choose architecture: layers, activation, dropout
*   Implement preprocessing pipeline for training

**Files:**
*   `cable_model.py`
*   Preprocessing parts inside `train_cable_failure_model.py`

### 3. Moataz & Khaled — Deep Learning Training Engineers
**Responsibilities:**
*   Implement training loop
*   Loss Function: CrossEntropyLoss
*   Optimizer: Adam
*   Track training + validation loss
*   Save model (`model_cable_failure.pt`)
*   Generate Curves: Training vs Validation Loss, Training Accuracy Curve
*   Generate Confusion Matrix

**Files:**
*   `train_cable_failure_model.py`
*   `plots/training_validation_curve.png`
*   `plots/training_accuracy_curve.png`
*   `confusion_matrix.png`
*   `model_cable_failure.pt`

### 4. Yusuf  — Backend Engineer (FastAPI)
**Responsibilities:**
*   Build FastAPI prediction server
*   Endpoints: `/predict_single`, `/predict_bulk`
*   Input validation
*   Load model + scaling pipeline
*   Fallback logic if PyTorch fails
*   Return JSON responses for frontend

**Files:**
*   `backend/main.py`
*   `backend/model_utils.py`

### 5. Mohamed — Frontend Engineer (React + Vite)
**Responsibilities:**
*   Full UI design
*   Pages: Home, Single Prediction, Bulk Upload, Model Performance
*   File Upload system
*   Charts (bar/pie)
*   Pagination + Filtering
*   Connect frontend to API

**Files:**
*   `frontend/src/App.tsx`
*   `frontend/src/components/*`
*   `frontend/src/utils/logic.ts`
*   `frontend/src/config/api.ts`

### 6. Tayam — Integration + Deployment Manager
**Responsibilities:**
*   Connect frontend ↔ backend
*   Build environment setup scripts
*   Ensure smooth local deployment
*   Write README.md
*   Upload final project to GitHub repo

**Files:**
*   `run_training.bat`
*   `setup_and_run.bat`
*   `README.md`
*   `project_summary.md`

---

## 📌 File Structure

```
project/
│── backend/
│   │── main.py
│   │── model_utils.py
│   │── requirements.txt
│
│── frontend/
│   │── src/
│   │   │── components/
│   │   │── config/
│   │   │── utils/
│   │   │── App.tsx
│   │── public/
│   │── package.json
│
│── plots/
│   │── training_validation_curve.png
│   │── training_accuracy_curve.png
│
│── tools/
│── cable_model.py
│── train_cable_failure_model.py
│── model_cable_failure.pt
│── Cable_Failure_Prediction.ipynb
│── run_training.bat
│── setup_and_run.bat
│── X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl
│── README.md
```

---

## 📌 Model Logic
*   **Input**: 6 features (Age, Partial Discharge, Neutral Corrosion, Loading, Visual Condition)
*   **Architecture**: MLP (Multi-Layer Perceptron) → 3-layer neural network
*   **Output**: 3 classes
*   **Mapping**:
    *   `0` → **Low Risk**
    *   `1` → **Medium Risk**
    *   `2` → **High Risk**
*   **Labeling**: Health Index is used as the target label for supervised learning.

---

## 📌 Running Instructions

To retrain the model and generate new plots:
```sh
.\run_training.bat
```

To run the full application (Backend + Frontend):
1.  **Start Backend**:
    ```sh
    cd backend
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
    ```
2.  **Start Frontend**:
    ```sh
    cd frontend
    npm run dev
    ```



