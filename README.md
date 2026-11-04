# T20 Cricket Score Prediction

A machine learning project that predicts T20 cricket match scores from match
conditions. It compares three regression models and uses KMeans clustering to
look for natural groupings in the data. Built for CSE422 (Artificial
Intelligence) at BRAC University.

## Dataset
- 1,500 matches, 9 features
- Target: `Predicted Score` (continuous, so this is a regression problem)
- Features: Overs Played, Wickets Lost, Run Rate, Opponent Strength,
  Home/Away, Pitch Condition, Weather (plus Match ID)

## Approach
- **EDA:** correlation heatmaps, score distribution (right-skewed), and
  relationships between score and overs, run rate, pitch, and venue
- **Preprocessing:** no missing values, one-hot encoding for categorical
  features, Min-Max scaling for numerical features
- **Split:** 80% train / 20% test (random state 42)
- **Models:** Linear Regression, Decision Tree Regressor, Neural Network
  (MLPRegressor), plus KMeans clustering (elbow method, PCA visualization)

## Results
| Model | R² | RMSE |
|-------|-----|------|
| Linear Regression | 0.913 | 21.010 |
| Neural Network (MLP) | 0.950 | 15.922 |
| Decision Tree | 0.958 | 14.754 |

The Decision Tree performed best, suggesting the relationship between match
conditions and score is non-linear. Overs played and run rate were the
strongest predictors.

## Built With
Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn

## How to Run
```
pip install -r requirements.txt
jupyter notebook cricket_score_prediction.ipynb
```
Keep `t20_cricket_match_score_prediction.csv` in the same folder as the notebook.

## Team
Oishee (me), Pushpita
