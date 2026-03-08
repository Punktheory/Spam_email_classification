# Spam Email Classification with CNN

A deep learning project that classifies emails as **spam** or **ham (not spam)** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Overview

This project was completed as Programming Assignment 2 for COMP2211. It explores how CNNs can be applied to Natural Language Processing (NLP) to detect spam emails — a real-world problem where traditional rule-based filters often fail against evolving spam tactics.

---

## Pipeline

### Section 1 — Preprocessing

| Step | Description |
|------|-------------|
| **Step 0 – Clean Emails** | Lowercase, remove HTML tags, URLs, punctuation, and special characters |
| **Step 1 – Build Vocabulary** | Keep only words appearing **≥5 times**; map each to a unique integer index |
| **Step 2 – Encode Emails** | Convert each email to a sequence of integers; unknown words → `0` |
| **Step 3 – Pad Sequences** | Pad or truncate all sequences to a fixed length of **300 tokens** |

### Section 2 — Model

**Architecture:**

- **Optimizer:** Adam | **Loss:** Binary Crossentropy | **Epochs:** 25 | **Batch size:** 32  
- **Test accuracy:** >90% with ~2 million parameters

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | CNN model |
| NumPy | Array operations & padding |
| scikit-learn | Train/test split |
| HuggingFace `datasets` | Dataset loading |
| Python `re`, `html` | Text cleaning |

---

## Dataset

`full_spam_dataset.json` — 1000 labelled emails (shuffled, `seed=42`)  
- `label = 0` → Ham  
- `label = 1` → Spam

---

## How to Run

1. Open the notebook in **Google Colab**  
   `Runtime → Change runtime type → GPU`
2. Upload `full_spam_dataset.json` into the `sample_data/` folder
3. Run all cells top to bottom

---

## Key Concepts

- Text preprocessing & tokenization for NLP
- Word embeddings for semantic representation
- 1D Convolutional layers for local pattern detection in text
- Dropout regularization to prevent overfitting
- Binary classification with sigmoid activation
