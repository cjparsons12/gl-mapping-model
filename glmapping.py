import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import joblib
import argparse

class NLPredictor:
    def __init__(self):
        """Initialize the NLP predictor with empty models and vectorizers."""
        self.pipeline = None
        self.trained = False
    
    def train(self, csv_path, input_cols, output_cols):
        """
        Train the model using data from a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file
            input_cols (list): List of column names containing natural language input
            output_cols (list): List of column names containing integer output values
        """
        # Load the data
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Check if all columns exist
        for col in input_cols + output_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the CSV file")
        
        # Prepare inputs (concatenate the text columns if there are multiple)
        X_text = df[input_cols[0]].astype(str)
        if len(input_cols) > 1:
            for col in input_cols[1:]:
                X_text = X_text + " " + df[col].astype(str)
        
        # Prepare outputs
        y = df[output_cols].astype(int)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=42
        )
        
        print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
        
        # Create a pipeline with TF-IDF vectorizer and Random Forest regressor
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
        ])
        
        # Train the model
        print("Training the model (this might take a while)...")
        self.pipeline.fit(X_train, y_train)
        self.trained = True
        
        # Evaluate the model
        train_score = self.pipeline.score(X_train, y_train)
        test_score = self.pipeline.score(X_test, y_test)
        print(f"Training R² score: {train_score:.4f}")
        print(f"Testing R² score: {test_score:.4f}")
        
        # Predict on test data and calculate MAE for each output
        y_pred = self.pipeline.predict(X_test)
        mae = np.mean(np.abs(y_test.values - y_pred), axis=0)
        for i, col in enumerate(output_cols):
            print(f"MAE for {col}: {mae[i]:.2f}")
        
        return test_score
    
    def predict(self, text1, text2=None):
        """
        Make predictions for new text inputs.
        
        Args:
            text1 (str): First natural language input
            text2 (str, optional): Second natural language input
            
        Returns:
            numpy.ndarray: Predicted integer values (rounded)
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet")
        
        # Combine inputs if both are provided
        if text2 is not None:
            input_text = f"{text1} {text2}"
        else:
            input_text = text1
        
        # Make predictions
        predictions = self.pipeline.predict([input_text])
        
        # Round to integers
        rounded_predictions = np.round(predictions).astype(int)
        
        return rounded_predictions[0]
    
    def save(self, model_path):
        """Save the trained model to a file."""
        if not self.trained:
            raise ValueError("Cannot save an untrained model")
        
        joblib.dump(self.pipeline, model_path)
        print(f"Model saved to {model_path}")
    
    def load(self, model_path):
        """Load a trained model from a file."""
        self.pipeline = joblib.load(model_path)
        self.trained = True
        print(f"Model loaded from {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and predict with NLP model")
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--csv', required=True, help='Path to the CSV file')
    train_parser.add_argument('--input-cols', required=True, nargs='+', 
                             help='Column names for natural language input')
    train_parser.add_argument('--output-cols', required=True, nargs='+', 
                             help='Column names for integer output')
    train_parser.add_argument('--model-path', default='nlp_model.joblib',
                             help='Path to save the trained model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model-path', default='nlp_model.joblib',
                               help='Path to the trained model')
    predict_parser.add_argument('--text1', required=True, help='First natural language input')
    predict_parser.add_argument('--text2', help='Second natural language input (optional)')
    
    # Batch predict command
    batch_parser = subparsers.add_parser('batch', help='Make batch predictions on a CSV')
    batch_parser.add_argument('--model-path', default='nlp_model.joblib',
                             help='Path to the trained model')
    batch_parser.add_argument('--csv', required=True, help='Path to input CSV file')
    batch_parser.add_argument('--input-cols', required=True, nargs='+', 
                             help='Column names for natural language input')
    batch_parser.add_argument('--output-csv', required=True,
                             help='Path to save prediction results')
    
    args = parser.parse_args()
    
    predictor = NLPredictor()
    
    if args.command == 'train':
        predictor.train(args.csv, args.input_cols, args.output_cols)
        predictor.save(args.model_path)
    
    elif args.command == 'predict':
        predictor.load(args.model_path)
        predictions = predictor.predict(args.text1, args.text2)
        print("Predictions:", predictions)
    
    elif args.command == 'batch':
        predictor.load(args.model_path)
        
        # Load data
        df = pd.read_csv(args.csv)
        
        # Prepare inputs
        X_text = df[args.input_cols[0]].astype(str)
        if len(args.input_cols) > 1:
            for col in args.input_cols[1:]:
                X_text = X_text + " " + df[col].astype(str)
        
        # Make predictions
        all_predictions = []
        for text in X_text:
            pred = predictor.predict(text)
            all_predictions.append(pred)
        
        # Convert to DataFrame
        pred_df = pd.DataFrame(all_predictions, 
                              columns=[f'predicted_{i}' for i in range(len(all_predictions[0]))])
        
        # Combine with original data
        result_df = pd.concat([df, pred_df], axis=1)
        
        # Save results
        result_df.to_csv(args.output_csv, index=False)
        print(f"Batch predictions saved to {args.output_csv}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
