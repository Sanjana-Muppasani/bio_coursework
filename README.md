# Biomedical Research: AI & Machine Learning Workflows

This repository contains the analysis and machine learning workflows for five distinct biomedical research questions, ranging from protein sequence homology to clinical disease prediction and medical imaging.

## Project Overview

This project explores the application of various AI techniques—including **Network Analysis**, **Logistic Regression**, **Ensemble Learning**, and **Deep Learning**—to solve complex biomedical challenges.

### Research Questions

1. **Proteomics:** Do protein sequences form distinct communities of homologues?
2. **Clinical Prediction:** Is it possible to predict the worsening of dementia ratings after an initial visit?
3. **Operations:** Which combination of algorithms best predicts patient appointment "no-shows"?
4. **Functional Genomics:** How well can we predict protein function directly from amino acid sequences?
5. **Ophthalmology:** Can deep neural networks effectively classify retinal images to identify eye disorders?

---

## Methodology & Technical Workflow

The project is divided into specialized pipelines for different data types:

### 1. Sequence Similarity Networks (SSN)

* **Alignment:** Uses the **Needleman-Wunsch** global alignment algorithm with a **BLOSUM62** substitution matrix to capture evolutionary relationships.
* **Graph Construction:** Sequences are parsed into undirected graphs where edges represent distance thresholds.
* **Analysis:** Employs the **Infomap** algorithm for community detection and analyzes topology via density, diameter, and transitivity.

### 2. Clinical Predictive Modeling

* **Dementia Progression:** Utilizes **Logistic Regression** with L1/L2 regularization to predict worsening Clinical Dementia Rating (CDR) scores.
* **No-Show Prediction:** Implements a comparative study of **KNN**, **Random Forest**, and **Gradient Boosting**.
* **Ensemble Methods:** Explores **Stacked Classifiers** and **Weighted Soft Voting** to improve robustness against imbalanced datasets.

### 3. Deep Learning Architectures

* **Protein Function:** Benchmarks **1D-CNN**, **LSTM**, and **GRU** models. The 1D-CNN was optimized using the **Hyperband** algorithm via Keras Tuner.
* **Retinal Imaging:** Compares a **Custom CNN** against a **Transfer Learning** approach using **MobileNetV2**.
* **Interpretability:** Uses **SHAP** values to visualize which biological features in OCT scans influence model predictions.

---

## Key Findings

| Research Task | Best Performing Model | Key Metric | Result |
| --- | --- | --- | --- |
| **Dementia Prediction** | Logistic Regression (L1) | Accuracy | 79.33% (Note: Failed to predict minority class) |
| **No-Show Prediction** | Gradient Boosting | Precision | 0.55 |
| **Protein Function** | 1D-CNN | Accuracy | 83.22% |
| **Retinal Imaging** | MobileNetV2 | Weighted F1 | 90% |

### Summary of Observations

* **Protein Structure:** Sequence analysis confirms that proteins organize into distinct homologous communities rather than a uniform continuum.
* **Class Imbalance:** Traditional models like Logistic Regression struggled significantly with imbalanced clinical data, defaulting to majority class predictions despite high accuracy.
* **Deep Learning:** CNNs outperformed RNNs in protein function prediction, suggesting that spatial pattern detection is more critical than temporal memory for this task.
* **Medical Imaging:** While custom CNNs are effective for binary "normal vs. diseased" screening, Transfer Learning (MobileNetV2) is required for accurate multi-class disease identification.

---

## Requirements & Setup

* **Languages:** Python
* **Libraries:** BioPython (PairwiseAligner), Izma (Data loading), Keras/TensorFlow (Deep Learning), Scikit-learn (Machine Learning), Keras Tuner (Hyperparameter optimization).

> **Hardware Note:** Images were resized to  to accommodate computational constraints during training.

---