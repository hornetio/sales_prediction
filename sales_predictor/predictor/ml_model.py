import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from io import StringIO


class SalesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None

    def load_model(self, model_path='predictor/model.pkl'):
        self.model = joblib.load(model_path)

    def load_scaler(self, scaler_path='predictor/scaler.pkl'):
        self.scaler = joblib.load(scaler_path)

    def predict(self, data):
        if self.model is None or self.scaler is None:
            raise ValueError("Model or Scaler not loaded.")

        data_scaled = self.scaler.transform(data)
        predictions = self.model.predict(data_scaled)
        return predictions

    def fit(self, X_train, y_train):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        self.scaler = StandardScaler()
        self.scaler.fit(X_train)

        # Save model and scaler
        joblib.dump(self.model, 'predictor/model.pkl')
        joblib.dump(self.scaler, 'predictor/scaler.pkl')

    def save_predictions_to_csv(self, predictions, output_file='predictor/predictions.csv'):
        df = pd.DataFrame(predictions, columns=['Predicted_Sales'])
        df.to_csv(output_file, index=False)
