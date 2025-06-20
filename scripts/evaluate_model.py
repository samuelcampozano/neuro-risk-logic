#!/usr/bin/env python3
"""
Comprehensive model evaluation with multiple metrics and visualizations.
Supports comparing multiple model versions and generating reports.
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
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold

from app.config import settings
from app.utils.feature_definitions import load_feature_definitions
from loguru import logger

# Configure logging
logger.add(
    "logs/model_evaluation.log",
    rotation="10 MB",
    level="INFO"
)


class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to model file to evaluate
        """
        self.model_path = model_path or settings.get_model_path()
        self.model = None
        self.metadata = {}
        self.feature_definitions = load_feature_definitions()
        self.evaluation_results = {}
        
    def load_model(self) -> bool:
        """Load model and metadata."""
        try:
            self.model = joblib.load(self.model_path)
            
            # Load metadata
            metadata_path = self.model_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            logger.info(f"Loaded model from {self.model_path}")
            logger.info(f"Model version: {self.metadata.get('version', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def load_test_data(self, test_file: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test data for evaluation.
        
        Args:
            test_file: Optional test data file
            
        Returns:
            Tuple of (X_test, y_test)
        """
        if test_file:
            # Load specific test file
            df = pd.read_csv(test_file)
        else:
            # Load most recent synthetic data and use last 20%
            synthetic_files = list(settings.synthetic_data_dir.glob("synthetic_data_*.csv"))
            if not synthetic_files:
                raise ValueError("No test data available")
            
            latest_file = max(synthetic_files, key=lambda p: p.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Use last 20% as test set
            test_size = int(len(df) * 0.2)
            df = df.tail(test_size)
        
        # Prepare features
        feature_cols = [
            'consanguinity', 'family_neuro_history', 'seizures_history',
            'brain_injury_history', 'psychiatric_diagnosis', 'substance_use',
            'suicide_ideation', 'psychotropic_medication', 'birth_complications',
            'extreme_poverty', 'education_access_issues', 'healthcare_access',
            'disability_diagnosis', 'breastfed_infancy', 'violence_exposure'
        ]
        
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
            features.append(gender_mapping.get(row.get('gender', 'Other'), 2))
            
            X_list.append(features)
        
        X = np.array(X_list)
        y = (df['risk_score'] >= 0.5).astype(int).values
        
        logger.info(f"Loaded test data: {X.shape[0]} samples")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def evaluate_predictions(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = cm[0, 0]
        metrics['false_positives'] = cm[0, 1]
        metrics['false_negatives'] = cm[1, 0]
        metrics['true_positives'] = cm[1, 1]
        
        # Calculate additional metrics
        metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        metrics['negative_predictive_value'] = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0
        
        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist()
        }
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        metrics['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        }
        
        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = roc_thresholds[optimal_idx]
        metrics['optimal_threshold'] = float(optimal_threshold)
        
        # Performance at optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        metrics['optimal_threshold_metrics'] = {
            'accuracy': accuracy_score(y_test, y_pred_optimal),
            'precision': precision_score(y_test, y_pred_optimal, zero_division=0),
            'recall': recall_score(y_test, y_pred_optimal, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_optimal, zero_division=0)
        }
        
        return metrics
    
    def evaluate_calibration(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model calibration.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Calibration metrics
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10, strategy='uniform'
        )
        
        calibration_metrics = {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist(),
            'expected_calibration_error': np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        }
        
        return calibration_metrics
    
    def evaluate_feature_importance(self) -> Dict[str, float]:
        """
        Analyze feature importance.
        
        Returns:
            Feature importance dictionary
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        feature_names = [
            'consanguinity', 'family_neuro_history', 'seizures_history',
            'brain_injury_history', 'psychiatric_diagnosis', 'substance_use',
            'suicide_ideation', 'psychotropic_medication', 'birth_complications',
            'extreme_poverty', 'education_access_issues', 'healthcare_access',
            'disability_diagnosis', 'breastfed_infancy', 'violence_exposure',
            'social_support_encoded', 'age_normalized', 'gender_encoded'
        ]
        
        importance_dict = {}
        for idx, importance in enumerate(self.model.feature_importances_):
            if idx < len(feature_names):
                importance_dict[feature_names[idx]] = float(importance)
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Cross-validation results
        """
        cv_scores = {}
        
        # Multiple scoring metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for metric in scoring_metrics:
            scores = cross_val_score(
                self.model, X, y, 
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                scoring=metric
            )
            cv_scores[metric] = {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'scores': scores.tolist()
            }
        
        return cv_scores
    
    def create_evaluation_plots(self, X_test: np.ndarray, y_test: np.ndarray, output_dir: Path = None):
        """
        Create comprehensive evaluation plots.
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save plots
        """
        if output_dir is None:
            output_dir = settings.models_dir / "evaluation_plots"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=300)
        plt.close()
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = np.mean(precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300)
        plt.close()
        
        # 3. Confusion Matrix Heatmap
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low Risk', 'High Risk'],
                   yticklabels=['Low Risk', 'High Risk'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        
        # 4. Feature Importance Plot
        feature_importance = self.evaluate_feature_importance()
        if feature_importance:
            plt.figure(figsize=(10, 8))
            features = list(feature_importance.keys())[:15]  # Top 15
            importances = [feature_importance[f] for f in features]
            
            y_pos = np.arange(len(features))
            plt.barh(y_pos, importances, color='skyblue')
            plt.yticks(y_pos, features)
            plt.xlabel('Importance')
            plt.title('Top 15 Feature Importances')
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_importance.png', dpi=300)
            plt.close()
        
        # 5. Calibration Plot
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'calibration_plot.png', dpi=300)
        plt.close()
        
        # 6. Score Distribution
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Low Risk', color='green')
        plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='High Risk', color='red')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Score Distribution by Class')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.boxplot([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]], 
                   labels=['Low Risk', 'High Risk'])
        plt.ylabel('Predicted Probability')
        plt.title('Score Distribution Boxplot')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'score_distribution.png', dpi=300)
        plt.close()
        
        logger.info(f"Saved evaluation plots to {output_dir}")
    
    def generate_report(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Complete evaluation report
        """
        logger.info("Generating comprehensive evaluation report...")
        
        # Basic evaluation
        metrics = self.evaluate_predictions(X_test, y_test)
        
        # Calibration
        calibration = self.evaluate_calibration(X_test, y_test)
        
        # Feature importance
        feature_importance = self.evaluate_feature_importance()
        
        # Cross-validation
        cv_results = self.cross_validate(X_test, y_test)
        
        # Classification report
        y_pred = self.model.predict(X_test)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Compile full report
        report = {
            'model_info': {
                'path': str(self.model_path),
                'version': self.metadata.get('version', 'unknown'),
                'training_date': self.metadata.get('training_date', 'unknown'),
                'model_type': self.metadata.get('model_type', 'unknown')
            },
            'evaluation_date': datetime.now().isoformat(),
            'test_set_size': len(X_test),
            'test_set_balance': {
                'low_risk': int(np.sum(y_test == 0)),
                'high_risk': int(np.sum(y_test == 1))
            },
            'metrics': metrics,
            'calibration': calibration,
            'feature_importance': feature_importance,
            'cross_validation': cv_results,
            'classification_report': class_report
        }
        
        # Save report
        report_path = settings.models_dir / f"evaluation_report_{self.metadata.get('version', 'current')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved evaluation report to {report_path}")
        
        return report
    
    def compare_models(self, model_paths: List[Path]) -> pd.DataFrame:
        """
        Compare multiple model versions.
        
        Args:
            model_paths: List of model paths to compare
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        # Load test data once
        X_test, y_test = self.load_test_data()
        
        for model_path in model_paths:
            self.model_path = model_path
            if not self.load_model():
                continue
            
            # Evaluate model
            metrics = self.evaluate_predictions(X_test, y_test)
            
            # Add to comparison
            comparison_data.append({
                'version': self.metadata.get('version', 'unknown'),
                'training_date': self.metadata.get('training_date', 'unknown'),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'auc_roc': metrics['auc_roc'],
                'brier_score': metrics['brier_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('training_date', ascending=False)
        
        # Save comparison
        comparison_path = settings.models_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        return comparison_df


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate neurodevelopmental risk model"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model file (default: current model)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data file"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate evaluation plots"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all model versions"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for output files"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(Path(args.model) if args.model else None)
    
    if args.compare:
        # Compare all models
        logger.info("Comparing all model versions...")
        model_files = list(settings.models_dir.glob("model_v*.pkl"))
        
        if len(model_files) < 2:
            logger.warning("Need at least 2 models for comparison")
        else:
            comparison_df = evaluator.compare_models(model_files)
            print("\nModel Comparison:")
            print(comparison_df.to_string(index=False))
    else:
        # Evaluate single model
        if not evaluator.load_model():
            logger.error("Failed to load model")
            return
        
        # Load test data
        X_test, y_test = evaluator.load_test_data(args.test_data)
        
        # Generate report
        report = evaluator.generate_report(X_test, y_test)
        
        # Create plots if requested
        if args.plots:
            output_dir = Path(args.output_dir) if args.output_dir else None
            evaluator.create_evaluation_plots(X_test, y_test, output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"Model Version: {report['model_info']['version']}")
        print(f"Test Set Size: {report['test_set_size']} samples")
        print(f"Class Balance: {report['test_set_balance']}")
        
        print(f"\nPerformance Metrics:")
        metrics = report['metrics']
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1_score']:.3f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.3f}")
        print(f"  Brier Score: {metrics['brier_score']:.3f}")
        
        print(f"\nCross-Validation Results (5-fold):")
        for metric, cv_data in report['cross_validation'].items():
            print(f"  {metric}: {cv_data['mean']:.3f} ± {cv_data['std']:.3f}")
        
        print(f"\nTop 5 Important Features:")
        for i, (feature, importance) in enumerate(list(report['feature_importance'].items())[:5]):
            print(f"  {i+1}. {feature}: {importance:.3f}")
        
        print(f"\nOptimal Threshold: {metrics['optimal_threshold']:.3f}")
        print(f"Performance at Optimal Threshold:")
        opt_metrics = metrics['optimal_threshold_metrics']
        print(f"  Accuracy:  {opt_metrics['accuracy']:.3f}")
        print(f"  Precision: {opt_metrics['precision']:.3f}")
        print(f"  Recall:    {opt_metrics['recall']:.3f}")
        
        print("="*60)


if __name__ == "__main__":
    main()