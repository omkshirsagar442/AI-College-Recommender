# AI College Recommender

## Overview

AI College Recommender is a FastAPI and Machine Learning based backend system developed to recommend colleges based on student eligibility, entrance exam scores, category, and preferred branches.

The system processes college admission data and generates intelligent recommendations using Machine Learning techniques and filtering logic.

This repository contains the backend API, dataset processing, and ML recommendation engine used in the application.

---

# Features

- AI-based college recommendation system
- FastAPI backend integration
- Machine Learning powered ranking
- Multiple entrance exam support
- Branch-based filtering
- Category-based filtering
- Pagination support
- REST API architecture
- Lightweight backend system

---

# Technologies Used

## Backend
- Python
- FastAPI

## Machine Learning
- Scikit-learn
- RandomForestRegressor
- LabelEncoder

## Data Processing
- Pandas
- NumPy

---

# Repository Structure

```bash
AI-College-Recommender/
│
├── colleges.csv          # College admission dataset
├── main.py               # FastAPI backend and ML logic
├── requirements.txt      # Required Python libraries
└── README.md             # Project documentation
```

---

# Project Workflow

1. Load college dataset
2. Clean and preprocess data
3. Detect cutoff type automatically
4. Encode categorical values
5. Train Machine Learning model
6. Receive API request
7. Filter eligible colleges
8. Generate ML recommendation score
9. Return ranked college recommendations

---

# Machine Learning Model

The system uses a `RandomForestRegressor` model for recommendation scoring.

The model analyzes:
- Field
- Branch
- Category
- Cutoff score
- Location
- Admission year

The system then predicts and ranks eligible colleges.

---

# API Endpoints

## 1. Get Available Branches

### Endpoint

```http
GET /branches
```

### Example

```http
/branches?field=engineering
```

Returns all available branches for the selected field.

---

## 2. College Recommendation API

### Endpoint

```http
POST /recommend
```

### Request Body Example

```json
{
  "field": "engineering",
  "category": "open",
  "cutoff_value": 95,
  "exam_mode": "mhtcet",
  "branches": ["computer engineering"],
  "page": 1,
  "limit": 10
}
```

---

# Supported Exam Modes

- JEE
- MHT-CET
- MBA
- Medical
- GATE
- LLB
- Junior College
- Diploma
- Percentage Based Admission

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/AI-College-Recommender.git
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Backend Server

```bash
uvicorn main:app --reload
```

---

# Output

The API returns:
- Eligible colleges
- Recommended branches
- Cutoff information
- College location
- Admission year
- Ranked recommendation results

---

# Applications

- College recommendation systems
- Student counseling platforms
- Admission guidance systems
- Educational analytics
- AI-based career guidance platforms

---

# Advantages

- Fast recommendation generation
- Smart ML-based filtering
- Supports multiple exam types
- Scalable backend architecture
- REST API integration
- Easy frontend integration

---

# Future Improvements

- Authentication system
- Database integration
- Cloud deployment
- Advanced recommendation algorithms
- Personalized recommendations
- Frontend dashboard integration

---

# Purpose of Repository

This repository serves as the Machine Learning and FastAPI backend module for intelligent college recommendation and admission prediction.

---

