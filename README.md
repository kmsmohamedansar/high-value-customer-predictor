# High-Value Customer Predictor

Is this customer high-value? A model gives an answer, and, more importantly, shows why.

**Try it:** [live demo on Hugging Face Spaces](https://huggingface.co/spaces/kmsmohamedansar/high-value-customer-predictor)

## Why it explains itself

A model that says "high value" with no reason attached isn't much use to a team that has to act on it. So this one comes with an explanation. A Random Forest makes the prediction, and a SHAP chart shows which inputs pushed the answer one way or the other.

## What you put in

- **Recency:** days since the last purchase
- **Frequency:** how many past orders
- **Average order value**
- **Total profit**
- **Average days between orders**
- **Region:** Central, East, South or West
- **Segment:** Consumer, Corporate or Home Office

## What you get back

- A prediction: high-value or not
- A confidence score
- A SHAP waterfall plot, so you can see what drove the result

## The model

It's a `RandomForestClassifier` trained on RFM style features (recency, frequency and money) from the Superstore dataset. These are the features it expects:

`recency_days`, `frequency`, `avg_order_value`, `total_profit`, `avg_days_between_orders`, `region_enc`, `segment_enc`

The front end is Streamlit, deployed on Hugging Face Spaces.

## Run it yourself

```bash
git clone https://github.com/kmsmohamedansar/high-value-customer-predictor.git
cd high-value-customer-predictor
pip install -r requirements.txt
streamlit run app.py
```
