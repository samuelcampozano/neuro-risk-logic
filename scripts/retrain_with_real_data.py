#!/usr/bin/env python3
"""
Retrain ML model using real clinical data and database assessments.
Implements incremental learning and model versioning.
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
from typing import Dict, Any, Tuple, Optional, List
import shutil

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sqlalchemy import func

from app.config import settings
from app.database import SessionLocal
from app.models.assessment import Assessment
from app.models.predictor import get_predictor
from scripts.train_model import ModelTrainer
from scripts.preprocess_real_data import RealDataPreprocessor
from loguru import logger

# Configure logging
logger.add("logs/model_retraining.log", rotation="10 MB", retention="30 days", level="INFO")


class IncrementalModelTrainer:
    """Handles incremental model training with real data."""

    def __init__(self, min_new_samples: int = 50):
        """
        Initialize incremental trainer.

        Args:
            min_new_samples: Minimum new samples needed for retraining
        """
        self.min_new_samples = min_new_samples
        self.current_model = None
        self.model_metadata = {}
        self.training_history = []

        # Load current model
        try:
            predictor = get_predictor()
            if predictor.is_loaded:
                self.current_model = predictor.model
                self.model_metadata = predictor.model_metadata
                logger.info(
                    f"Loaded current model version: {self.model_metadata.get('version', 'unknown')}"
                )
        except Exception as e:
            logger.warning(f"Could not load current model: {e}")

    def collect_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Collect data from multiple sources:
        1. Synthetic baseline data
        2. Real clinical data (if available)
        3. User-submitted assessments from database

        Returns:
            Tuple of (synthetic_df, real_df, database_df)
        """
        # 1. Load synthetic baseline
        synthetic_files = list(settings.synthetic_data_dir.glob("synthetic_data_*.csv"))
        synthetic_df = None

        if synthetic_files:
            latest_synthetic = max(synthetic_files, key=lambda p: p.stat().st_mtime)
            synthetic_df = pd.read_csv(latest_synthetic)
            logger.info(
                f"Loaded {len(synthetic_df)} synthetic samples from {latest_synthetic.name}"
            )

        # 2. Load preprocessed real data
        real_files = list(settings.synthetic_data_dir.glob("processed_real_data_*.csv"))
        real_df = None

        if real_files:
            latest_real = max(real_files, key=lambda p: p.stat().st_mtime)
            real_df = pd.read_csv(latest_real)
            logger.info(f"Loaded {len(real_df)} real samples from {latest_real.name}")

        # 3. Load database assessments
        db = SessionLocal()
        database_df = None

        try:
            # Get assessments not used in training before
            assessments = db.query(Assessment).filter(Assessment.consent_given == True).all()

            if assessments:
                # Convert to DataFrame
                data = []
                for assessment in assessments:
                    record = {
                        "age": assessment.age,
                        "gender": assessment.gender,
                        "consanguinity": assessment.consanguinity,
                        "family_neuro_history": assessment.family_neuro_history,
                        "seizures_history": assessment.seizures_history,
                        "brain_injury_history": assessment.brain_injury_history,
                        "psychiatric_diagnosis": assessment.psychiatric_diagnosis,
                        "substance_use": assessment.substance_use,
                        "suicide_ideation": assessment.suicide_ideation,
                        "psychotropic_medication": assessment.psychotropic_medication,
                        "birth_complications": assessment.birth_complications,
                        "extreme_poverty": assessment.extreme_poverty,
                        "education_access_issues": assessment.education_access_issues,
                        "healthcare_access": assessment.healthcare_access,
                        "disability_diagnosis": assessment.disability_diagnosis,
                        "social_support_level": assessment.social_support_level,
                        "breastfed_infancy": assessment.breastfed_infancy,
                        "violence_exposure": assessment.violence_exposure,
                        "risk_score": assessment.risk_score,
                        "risk_level": assessment.risk_level,
                        "assessment_date": assessment.assessment_date,
                    }
                    data.append(record)

                database_df = pd.DataFrame(data)
                logger.info(f"Loaded {len(database_df)} assessments from database")

        except Exception as e:
            logger.error(f"Error loading database assessments: {e}")
        finally:
            db.close()

        return synthetic_df, real_df, database_df

    def combine_datasets(
        self,
        synthetic_df: Optional[pd.DataFrame],
        real_df: Optional[pd.DataFrame],
        database_df: Optional[pd.DataFrame],
        weights: Dict[str, float] = None,
    ) -> pd.DataFrame:
        """
        Intelligently combine datasets with appropriate weighting.

        Args:
            synthetic_df: Synthetic training data
            real_df: Real clinical data
            database_df: User-submitted assessments
            weights: Weight for each data source

        Returns:
            Combined DataFrame ready for training
        """
        if weights is None:
            weights = {
                "synthetic": 0.3,  # Lower weight for synthetic
                "real": 0.5,  # Higher weight for real clinical
                "database": 0.2,  # Moderate weight for user submissions
            }

        datasets = []

        # Add synthetic data (downsampled based on weight)
        if synthetic_df is not None and len(synthetic_df) > 0:
            n_synthetic = int(len(synthetic_df) * weights["synthetic"])
            synthetic_sample = synthetic_df.sample(
                n=min(n_synthetic, len(synthetic_df)), random_state=42
            )
            synthetic_sample["data_source"] = "synthetic"
            datasets.append(synthetic_sample)

        # Add real clinical data
        if real_df is not None and len(real_df) > 0:
            real_df["data_source"] = "real_clinical"
            datasets.append(real_df)

        # Add database assessments
        if database_df is not None and len(database_df) > 0:
            database_df["data_source"] = "user_submitted"
            datasets.append(database_df)

        if not datasets:
            raise ValueError("No data available for training")

        # Combine all datasets
        combined_df = pd.concat(datasets, ignore_index=True)

        # Log data composition
        source_counts = combined_df["data_source"].value_counts()
        logger.info("Combined dataset composition:")
        for source, count in source_counts.items():
            logger.info(f"  {source}: {count} samples ({count/len(combined_df)*100:.1f}%)")

        return combined_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training.

        Args:
            df: Combined DataFrame

        Returns:
            Tuple of (X, y) arrays
        """
        # Feature columns
        feature_cols = [
            "consanguinity",
            "family_neuro_history",
            "seizures_history",
            "brain_injury_history",
            "psychiatric_diagnosis",
            "substance_use",
            "suicide_ideation",
            "psychotropic_medication",
            "birth_complications",
            "extreme_poverty",
            "education_access_issues",
            "healthcare_access",
            "disability_diagnosis",
            "breastfed_infancy",
            "violence_exposure",
        ]

        X_list = []

        for _, row in df.iterrows():
            features = []

            # Binary features
            for col in feature_cols:
                features.append(float(row[col]))

            # Social support encoding
            social_mapping = {"isolated": 0, "moderate": 1, "supported": 2}
            features.append(social_mapping.get(row["social_support_level"], 1))

            # Age (normalized)
            features.append(row["age"] / 100.0)

            # Gender encoding
            gender_mapping = {"M": 0, "F": 1, "Other": 2}
            features.append(gender_mapping.get(row.get("gender", "Other"), 2))

            X_list.append(features)

        X = np.array(X_list)

        # Target variable (risk_score > 0.5)
        y = (df["risk_score"] >= 0.5).astype(int).values

        return X, y

    def train_incremental_model(
        self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train model with incremental learning strategy.

        Args:
            X: Feature matrix
            y: Target vector
            validation_split: Validation data percentage

        Returns:
            Training metrics
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        # Initialize new model with current model's parameters
        if self.current_model is not None:
            # Use warm start with existing trees
            new_model = RandomForestClassifier(
                n_estimators=self.current_model.n_estimators + 20,  # Add more trees
                max_depth=self.current_model.max_depth,
                min_samples_split=self.current_model.min_samples_split,
                min_samples_leaf=self.current_model.min_samples_leaf,
                class_weight="balanced",
                random_state=42,
                warm_start=True,
                n_jobs=-1,
            )

            # Copy existing trees if possible
            if hasattr(self.current_model, "estimators_"):
                new_model.estimators_ = self.current_model.estimators_
                new_model.n_estimators = len(new_model.estimators_)
        else:
            # Create new model from scratch
            new_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

        # Train model
        logger.info("Training incremental model...")
        new_model.fit(X_train, y_train)

        # Evaluate
        y_pred = new_model.predict(X_val)
        y_pred_proba = new_model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1_score": f1_score(y_val, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.0,
            "samples_trained": len(X_train),
            "samples_validated": len(X_val),
        }

        # Compare with current model
        if self.current_model is not None:
            current_pred_proba = self.current_model.predict_proba(X_val)[:, 1]
            current_auc = (
                roc_auc_score(y_val, current_pred_proba) if len(np.unique(y_val)) > 1 else 0.0
            )
            metrics["improvement"] = metrics["auc_roc"] - current_auc
            logger.info(f"Model improvement: {metrics['improvement']:.4f}")

        self.current_model = new_model
        return metrics

    def save_model_version(
        self, model: Any, metrics: Dict[str, Any], data_composition: Dict[str, int]
    ):
        """
        Save model with versioning and metadata.

        Args:
            model: Trained model
            metrics: Performance metrics
            data_composition: Data source counts
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{timestamp}"

        # Save model file
        model_path = settings.models_dir / f"model_{version}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")

        # Create metadata
        metadata = {
            "version": version,
            "training_date": datetime.now().isoformat(),
            "model_type": "RandomForestClassifier",
            "training_metrics": metrics,
            "data_composition": data_composition,
            "previous_version": self.model_metadata.get("version", "initial"),
            "incremental_training": True,
            "model_config": model.get_params(),
        }

        # Save metadata
        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update current model symlink
        current_model_path = settings.get_model_path()
        current_metadata_path = current_model_path.with_suffix(".json")

        # Backup current model
        if current_model_path.exists():
            backup_path = (
                settings.models_dir / f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
            shutil.copy2(current_model_path, backup_path)

        # Update current model
        joblib.dump(model, current_model_path)
        with open(current_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Updated current model to version {version}")

        # Add to training history
        self.training_history.append(
            {
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "data_composition": data_composition,
            }
        )

        # Save training history
        history_path = settings.models_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

    def should_retrain(self, database_df: Optional[pd.DataFrame]) -> bool:
        """
        Determine if model should be retrained based on new data.

        Args:
            database_df: New assessments from database

        Returns:
            True if retraining is recommended
        """
        if database_df is None or len(database_df) == 0:
            return False

        # Check if we have minimum new samples
        if len(database_df) < self.min_new_samples:
            logger.info(
                f"Only {len(database_df)} new samples, need {self.min_new_samples} for retraining"
            )
            return False

        # Check time since last training
        if self.model_metadata.get("training_date"):
            last_training = datetime.fromisoformat(self.model_metadata["training_date"])
            days_since = (datetime.now() - last_training).days

            if days_since < 7:  # Don't retrain more than weekly
                logger.info(f"Last training was {days_since} days ago, too recent")
                return False

        return True

    def run_retraining_pipeline(self, force: bool = False) -> Dict[str, Any]:
        """
        Run complete retraining pipeline.

        Args:
            force: Force retraining regardless of conditions

        Returns:
            Retraining results
        """
        # Collect data
        synthetic_df, real_df, database_df = self.collect_training_data()

        # Check if retraining is needed
        if not force and not self.should_retrain(database_df):
            logger.info("Retraining conditions not met, skipping")
            return {"status": "skipped", "reason": "insufficient_new_data"}

        # Combine datasets
        combined_df = self.combine_datasets(synthetic_df, real_df, database_df)

        # Get data composition
        data_composition = combined_df["data_source"].value_counts().to_dict()

        # Prepare features
        X, y = self.prepare_features(combined_df)
        logger.info(f"Prepared {len(X)} samples for training")
        logger.info(f"Class distribution: {np.bincount(y)}")

        # Train model
        metrics = self.train_incremental_model(X, y)

        # Log results
        logger.info("\nTraining Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall: {metrics['recall']:.3f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.3f}")

        if "improvement" in metrics:
            if metrics["improvement"] > 0:
                logger.info(f"  ✅ Model improved by {metrics['improvement']:.4f}")
            else:
                logger.warning(
                    f"  ⚠️ Model performance decreased by {abs(metrics['improvement']):.4f}"
                )

        # Save model if improved or forced
        if force or metrics.get("improvement", 0) >= -0.01:  # Allow 1% degradation
            self.save_model_version(self.current_model, metrics, data_composition)
            status = "success"
            message = "Model retrained and saved"
        else:
            logger.warning("Model performance degraded too much, keeping current model")
            status = "rejected"
            message = "Performance degradation exceeded threshold"

        return {
            "status": status,
            "message": message,
            "metrics": metrics,
            "data_composition": data_composition,
            "timestamp": datetime.now().isoformat(),
        }


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Retrain model with real clinical data")
    parser.add_argument(
        "--force", action="store_true", help="Force retraining regardless of conditions"
    )
    parser.add_argument(
        "--min-samples", type=int, default=50, help="Minimum new samples required for retraining"
    )
    parser.add_argument("--real-data", type=str, help="Path to new real clinical data file")

    args = parser.parse_args()

    # Process new real data if provided
    if args.real_data:
        logger.info(f"Processing new real data from {args.real_data}")
        preprocessor = RealDataPreprocessor(args.real_data)
        preprocessor.preprocess_data()
        preprocessor.save_processed_data()

    # Run retraining
    trainer = IncrementalModelTrainer(min_new_samples=args.min_samples)
    result = trainer.run_retraining_pipeline(force=args.force)

    # Print summary
    print("\n" + "=" * 60)
    print("RETRAINING SUMMARY")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")

    if "metrics" in result:
        print(f"\nPerformance Metrics:")
        for metric, value in result["metrics"].items():
            if metric != "improvement":
                print(f"  {metric}: {value:.3f}")

        if "improvement" in result["metrics"]:
            improvement = result["metrics"]["improvement"]
            if improvement > 0:
                print(f"\n✅ Model improved by {improvement:.4f}")
            else:
                print(f"\n⚠️ Model degraded by {abs(improvement):.4f}")

    if "data_composition" in result:
        print(f"\nData Sources Used:")
        for source, count in result["data_composition"].items():
            print(f"  {source}: {count} samples")

    print("=" * 60)


if __name__ == "__main__":
    main()
