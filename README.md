# 💎 High-Value Customer Predictor

This app uses customer purchase patterns to predict if a customer is **high-value** using a trained Random Forest classifier and provides **SHAP-based explanations** for interpretability.

[![Live on Hugging Face Spaces](https://img.shields.io/badge/demo-huggingface-orange?logo=huggingface)](https://huggingface.co/spaces/kmsmohamedansar/high-value-customer-predictor)

---

## 📊 Features

- Predicts whether a customer is high-value based on past behavior
- Built with **Streamlit**, **scikit-learn**, and **SHAP**
- Deployed live using **Hugging Face Spaces**
- User-friendly interface with visualization

---

## 🧠 Model Details

- Algorithm: RandomForestClassifier
- Trained on: Superstore dataset
- Features used:
  - `recency_days`
  - `frequency`
  - `avg_order_value`
  - `total_profit`
  - `avg_days_between_orders`
  - `region_enc`
  - `segment_enc`

---

## 🚀 Run Locally

1. Clone the repo:

```bash
git clone https://github.com/kmsmohamedansar/high-value-customer-predictor.git
cd high-value-customer-predictor
