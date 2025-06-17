#!/usr/bin/env python3
"""
Train machine learning model for neurodevelopmental risk assessment.
Supports training from synthetic data or database records.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from app.config import settings
from app.database import SessionLocal
from app.models.assessment import Assessment
from loguru import logger

# Configure logging
logger.add(
    "logs/model_training.log",
    rotation="10 MB",
    level="INFO"
)


class ModelTrainer:
    """Train and evaluate neurodevelopmental risk prediction model."""
    
    def __init__(self, model_version: str = None):
        """
        Initialize model trainer.
        
        Args:
            model_version: Version string for the model
        """
        self.model_version = model_version or f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model = None
        self.feature_names = []
        self.metrics = {}
        
        # Load feature definitions
        with open(settings.get_feature_definitions_path(), 'r') as f:
            self.feature_defs = json.load(f)
        
        logger.info(f"Initialized ModelTrainer version: {self.model_version}")
    
    def load_data_from_csv(self, csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data from CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Tuple of (X, y) arrays
        """
        logger.info(f"Loading data from {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Extract features
        feature_cols = []
        for feature in self.feature_defs['features']:
            if feature['name'] not in ['age', 'gender', 'social_support_level']:
                if feature['type'] == 'binary':
                    feature_cols.append(feature['name'])
        
        # Add encoded categorical features
        X_list = []
        
        for _, row in df.iterrows():
            features = []
            
            # Binary features
            for col in feature_cols:
                features.append(float(row[col]))
            
            # Social support encoding
            social_mapping = {'isolated': 0, 'moderate': 1, 'supported': 2}
            features.append(social_mapping.get(row['social_support_level'], 1))
            
            # Age (normalized)
            features.append(row['age'] / 100.0)
            
            # Gender encoding
            gender_mapping = {'M': 0, 'F': 1, 'Other': 2}
            features.append(gender_mapping.get(row['gender'], 2))
            
            X_list.append(features)
        
        X = np.array(X_list)
        
        # Extract targets (risk_score > 0.5 = 1)
        y = (df['risk_score'] >= 0.5).astype(int).values
        
        # Store feature names
        self.feature_names = feature_cols + ['social_support_encoded', 'age_normalized', 'gender_encoded']
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution: {np.bincount(y)}")
        
        return X, y
    
    def load_data_from_database(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data from database assessments.
        
        Returns:
            Tuple of (X, y) arrays
        """
        logger.info("Loading data from database")
        
        db = SessionLocal()
        try:
            assessments = db.query(Assessment).all()
            logger.info(f"Found {len(assessments)} assessments in database")
            
            if len(assessments) < 50:
                raise ValueError(f"Insufficient data: {len(assessments)} samples (minimum 50 required)")
            
            X_list = []
            y_list = []
            
            for assessment in assessments:
                # Extract features using the model's feature_vector property
                features = assessment.feature_vector
                X_list.append(features)
                
                # Target is high risk (risk_score >= 0.5)
                y_list.append(1 if assessment.risk_score >= 0.5 else 0)
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Set feature names
            self.feature_names = [f['name'] for f in self.feature_defs['features']]
            
            return X, y
            
        finally:
            db.close()
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=settings.random_seed, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Initialize model with configuration from feature definitions
        model_config = self.feature_defs.get('model_config', {})
        
        self.model = RandomForestClassifier(
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', 10),
            min_samples_split=model_config.get('min_samples_split', 5),
            min_samples_leaf=2,
            class_weight=model_config.get('class_weight', 'balanced'),
            random_state=settings.random_seed,
            n_jobs=-1
        )
        
        # Train model
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.random_seed),
            scoring='roc_auc'
        )
        
        self.metrics['cv_auc_mean'] = cv_scores.mean()
        self.metrics['cv_auc_std'] = cv_scores.std()
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.metrics['top_features'] = feature_importance.head(10).to_dict('records')
        
        logger.info("Model training completed")
        logger.info(f"Test Accuracy: {self.metrics['accuracy']:.3f}")
        logger.info(f"Test AUC-ROC: {self.metrics['auc_roc']:.3f}")
        logger.info(f"CV AUC: {self.metrics['cv_auc_mean']:.3f} ± {self.metrics['cv_auc_std']:.3f}")
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.metrics['confusion_matrix'] = cm.tolist()
        
        return self.metrics
    
    def save_model(self):
        """Save trained model and metadata."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Save model
        model_path = settings.get_model_path(self.model_version)
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save as current model
        current_model_path = settings.get_model_path()
        joblib.dump(self.model, current_model_path)
        logger.info(f"Model saved as current at {current_model_path}")
        
        # Save metadata
        metadata = {
            'version': self.model_version,
            'training_date': datetime.now().isoformat(),
            'model_type': 'RandomForestClassifier',
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'metrics': self.metrics,
            'model_config': self.model.get_params()
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        # Update current model metadata
        current_metadata_path = current_model_path.with_suffix('.json')
        with open(current_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_visualizations(self, output_dir: Path = None):
        """Create and save model performance visualizations."""
        if output_dir is None:
            output_dir = settings.models_dir / "visualizations"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature importance plot
        if 'top_features' in self.metrics:
            plt.figure(figsize=(10, 6))
            features_df = pd.DataFrame(self.metrics['top_features'])
            sns.barplot(data=features_df, x='importance', y='feature')
            plt.title('Top 10 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(output_dir / f'feature_importance_{self.model_version}.png')
            plt.close()
        
        # Confusion matrix
        if 'confusion_matrix' in self.metrics:
            plt.figure(figsize=(8, 6))
            cm = np.array(self.metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Low Risk', 'High Risk'],
                       yticklabels=['Low Risk', 'High Risk'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(output_dir / f'confusion_matrix_{self.model_version}.png')
            plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train neurodevelopmental risk assessment model"
    )
    parser.add_argument(
        "-s", "--source",
        choices=["csv", "database"],
        default="csv",
        help="Data source for training"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="CSV file path (if source is csv)"
    )
    parser.add_argument(
        "-v", "--version",
        type=str,
        default=None,
        help="Model version string"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization plots"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting model training")
    logger.info(f"Data source: {args.source}")
    
    # Initialize trainer
    trainer = ModelTrainer(model_version=args.version)
    
    # Load data
    if args.source == "csv":
        if not args.file:
            # Find most recent synthetic data file
            csv_files = list(settings.synthetic_data_dir.glob("synthetic_data_*.csv"))
            if not csv_files:
                raise ValueError("No synthetic data files found. Run generate_synthetic_data.py first.")
            csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Using most recent file: {csv_path}")
        else:
            csv_path = Path(args.file)
        
        X, y = trainer.load_data_from_csv(csv_path)
    else:
        X, y = trainer.load_data_from_database()
    
    # Train model
    metrics = trainer.train_model(X, y)
    
    # Save model
    trainer.save_model()
    
    # Create visualizations
    if args.visualize:
        trainer.create_visualizations()
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL TRAINING SUMMARY")
    print("="*60)
    print(f"Model version: {trainer.model_version}")
    print(f"Training samples: {len(X)}")
    print(f"\nPerformance Metrics:")
    print(f"  - Accuracy: {metrics['accuracy']:.3f}")
    print(f"  - Precision: {metrics['precision']:.3f}")
    print(f"  - Recall: {metrics['recall']:.3f}")
    print(f"  - F1-Score: {metrics['f1_score']:.3f}")
    print(f"  - AUC-ROC: {metrics['auc_roc']:.3f}")
    print(f"  - CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
    print(f"\nTop 5 Important Features:")
    for i, feature in enumerate(metrics['top_features'][:5]):
        print(f"  {i+1}. {feature['feature']}: {feature['importance']:.3f}")
    print("="*60)
    
    logger.info("Model training completed successfully")


if __name__ == "__main__":
    main()