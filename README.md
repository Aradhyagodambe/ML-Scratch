# Machine Learning from Scratch (NumPy Only)

This repository contains my personal implementations of core Machine Learning algorithms, built entirely from scratch using only standard Python and NumPy. 

I started this project primarily for my own personal understanding. By avoiding high-level frameworks like Scikit-Learn and building these algorithms from the ground up, my goal is to internalize the underlying mathematics, optimization techniques, and core logic behind them. I felt like sharing this journey, so I made the repository public in case others find it helpful to read through raw, framework-free code.

## Current Status and Future Plans

This repository is being actively built. I will continuously add more algorithms over time as my own understanding of the subject grows. 

Currently, this repository focuses strictly on classical Machine Learning algorithms. Once I have covered these fundamentals to my satisfaction, I plan to create a similar, separate repository dedicated entirely to Deep Learning models.

## Prerequisites

To effectively read, use, or contribute to this code, you will need:
* A solid prior understanding of Python.
* Familiarity with NumPy, specifically array manipulation, broadcasting, and basic linear algebra operations (like matrix multiplication).

## Algorithms Planned and Implemented

**Legend for tags `[Tier / Popularity]`**
*   **Tier:** **B** (Basic), **I** (Intermediate), **A** (Advanced)
*   **Popularity/Industry Usage:** **H** (High), **M** (Medium), **L** (Low/Niche)

### Supervised Learning
- [x] Linear Regression (OLS & Gradient Descent) `[B / H]`
- [x] Logistic Regression `[B / H]`
- [x] K-Nearest Neighbors (KNN) `[B / H]`
- [x] Naive Bayes `[B / H]`
- [x] Support Vector Machines (SVM) `[I / H]`
- [x] Decision Trees `[I / H]`
- [ ] Perceptron (Classical Linear Classifier) `[B / M]`
- [ ] Ridge, Lasso, and Elastic Net Regression `[I / H]`
- [ ] Linear & Quadratic Discriminant Analysis (LDA & QDA) `[I / M]`
- [ ] Gaussian Processes (GP) `[A / L]`

### Ensemble Learning
- [ ] Random Forest `[A / H]`
- [ ] AdaBoost & Gradient Boosting `[A / H]`
- [ ] XGBoost (Extreme Gradient Boosting) `[A / H]`

### Unsupervised Learning
- [ ] K-Means Clustering `[B / H]`
- [ ] Principal Component Analysis (PCA) `[B / H]`
- [ ] DBSCAN `[I / H]`
- [ ] Gaussian Mixture Models (GMM) `[A / M]`
- [ ] Hierarchical Agglomerative Clustering `[A / M]`
- [ ] Independent Component Analysis (ICA) `[A / L]`
- [ ] Truncated SVD / Matrix Factorization `[A / H]`
- [ ] t-SNE (t-Distributed Stochastic Neighbor Embedding) `[A / H]`

### Anomaly Detection
- [ ] Isolation Forest `[A / M]`

### Association Rule Learning
- [ ] Apriori Algorithm `[I / M]`

### Sequence Prediction
- [ ] Hidden Markov Models (HMM) `[A / L]`
