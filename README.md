# 💎 High-Value Customer Predictor

## In plain English

This project estimates whether a customer is likely to be especially valuable based on past purchasing behaviour. It also explains which behaviours influenced each prediction, so the result is easier to inspect instead of being a unexplained score.

**Status:** Working machine-learning demonstration using sample retail data.


This app uses customer purchase patterns to predict whether a customer is **high-value**, using a trained **Random Forest** classifier. It also provides **SHAP-based feature explanations** to interpret predictions.

👉 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/kmsmohamedansar/high-value-customer-predictor)**

---

## 📊 Features

- ✅ Predicts high-value customers based on their purchase behavior
- 📈 SHAP explainability for model transparency
- 🧠 Trained on Superstore-style customer data
- ⚡ Built with Streamlit and deployed to Hugging Face Spaces
- 🖥️ Clean and user-friendly UI

---

## 🧠 Model Details

- **Algorithm**: `RandomForestClassifier`
- **Trained on**: RFM-style features from Superstore dataset
- **Expected Features**:
  - `recency_days`
  - `frequency`
  - `avg_order_value`
  - `total_profit`
  - `avg_days_between_orders`
  - `region_enc`
  - `segment_enc`

---

## 🧾 Inputs & Outputs

### 📥 Inputs:
- **Recency** – Days since last purchase
- **Frequency** – Number of past orders
- **Average Order Value**
- **Total Profit**
- **Average Days Between Orders**
- **Region** – One of: Central, East, South, West
- **Segment** – One of: Consumer, Corporate, Home Office

### 📤 Outputs:
- Binary prediction: **High-Value** or **Not**
- Confidence score
- **SHAP waterfall plot** showing feature impact

---

## 🚀 Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/kmsmohamedansar/high-value-customer-predictor.git
   cd high-value-customer-predictor
