# MLOps project - House Price Prediction – From Scratch

## Description
This project was built from scratch with one goal:

To deeply understand how an ML system works end-to-end — from notebooks to a production-ready pipeline.

I rewrote every line on this project from zero, because I want to learn everything. (I didn't use AI, just as a tutor)
In the process I found inconsistencies or things that can be done better (e.g. use uv instead of pip), I try to implement best practices and understand why.

## Tech Stack
- Python
- scikit-learn
- MLflow
- Pandas / NumPy
- uv (dependency management)
- GitHub Actions (CI)

---

## Key Improvements
### 1. Dependency Management (uv)

Decided to use uv as the dependency manager instead of pip.

- Introduced dependency groups (training, inference, dev)
- Avoid installing unnecessary dependencies in each environment

**Result:**
- Better reproducibility
- Cleaner environments
- Lighter deployments

---

### 2. CI Pipeline Refactor (GitHub Actions)

Rebuilt the GitHub Actions pipeline in a more modular way and integrated uv.

- Structured pipeline from training → build → image publication
- Reduced coupling between steps
**Result:**
- More reliable CI
- Easier to maintain and extend

### 3. Removed price_per_sqft
Removed this feature because it depends directly on the target (price).
**Result:**
- Avoided data leakage
- More realistic model behavior

### 4. Feature Pipeline Fix
Some features were hardcoded in the pipeline.
Removed hardcoding
Standardized feature handling
**Result:**
- Avoids column mismatches
- Safer training and 

### 5. Feature Selection Consistency
Feature selection (RFE) was done in notebooks but not used in training.
Integrated selected features into the training pipeline
**Result:**
- Consistent experimentation and production

### 6. Feature Tracking (features.pkl)
Added a features.pkl file to improve image package.
**Result:**
- Full control over which features are used
- Easier reproducibility
- Prevents silent errors

### Pipeline Flow
1. Data processing
2. Feature engineering
3. Feature selection
4. Feature tracking (features.pkl)
5. Model training
6. CI pipeline → build → image publication

### Next Steps
Currently improving the project by building a frontend using Reflex (instead of Streamlit) to interact with the model.