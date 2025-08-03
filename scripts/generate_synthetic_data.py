#!/usr/bin/env python3
"""
Generate synthetic training data for neurodevelopmental risk assessment.
Uses logical rules based on clinical knowledge to create realistic data.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple

from app.config import settings
from loguru import logger

# Configure logging
logger.add("logs/data_generation.log", rotation="10 MB", level="INFO")


class SyntheticDataGenerator:
    """Generate synthetic neurodevelopmental assessment data."""

    def __init__(self, n_samples: int = 1000, random_seed: int = 42):
        """
        Initialize data generator.

        Args:
            n_samples: Number of samples to generate
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Load feature definitions
        with open(settings.get_feature_definitions_path(), "r") as f:
            self.feature_defs = json.load(f)

        logger.info(f"Initialized generator for {n_samples} samples")

    def generate_demographics(self) -> Tuple[int, str]:
        """Generate age and gender."""
        # Age distribution (skewed towards younger ages for developmental disorders)
        age = int(np.random.beta(2, 5) * 80 + 1)  # 1-80 years, skewed young

        # Gender distribution
        gender = np.random.choice(["M", "F", "Other"], p=[0.52, 0.47, 0.01])

        return age, gender

    def generate_risk_profile(self) -> str:
        """Randomly assign a risk profile to guide feature generation."""
        profiles = ["low_risk", "moderate_risk", "high_risk", "very_high_risk"]
        probabilities = [0.4, 0.35, 0.20, 0.05]
        return np.random.choice(profiles, p=probabilities)

    def generate_features(self, age: int, gender: str, risk_profile: str) -> Dict:
        """
        Generate features based on risk profile and demographics.

        Args:
            age: Subject age
            gender: Subject gender
            risk_profile: Risk category

        Returns:
            Dictionary of feature values
        """
        features = {"age": age, "gender": gender}

        # Base probabilities for each risk profile
        risk_probabilities = {
            "low_risk": {
                "consanguinity": 0.02,
                "family_neuro_history": 0.05,
                "seizures_history": 0.01,
                "brain_injury_history": 0.02,
                "psychiatric_diagnosis": 0.05,
                "substance_use": 0.03,
                "suicide_ideation": 0.01,
                "psychotropic_medication": 0.03,
                "birth_complications": 0.05,
                "extreme_poverty": 0.02,
                "education_access_issues": 0.03,
                "healthcare_access": 0.95,  # Protective
                "disability_diagnosis": 0.02,
                "breastfed_infancy": 0.85,  # Protective
                "violence_exposure": 0.02,
            },
            "moderate_risk": {
                "consanguinity": 0.05,
                "family_neuro_history": 0.20,
                "seizures_history": 0.05,
                "brain_injury_history": 0.08,
                "psychiatric_diagnosis": 0.25,
                "substance_use": 0.15,
                "suicide_ideation": 0.05,
                "psychotropic_medication": 0.20,
                "birth_complications": 0.15,
                "extreme_poverty": 0.10,
                "education_access_issues": 0.15,
                "healthcare_access": 0.70,
                "disability_diagnosis": 0.10,
                "breastfed_infancy": 0.60,
                "violence_exposure": 0.15,
            },
            "high_risk": {
                "consanguinity": 0.15,
                "family_neuro_history": 0.40,
                "seizures_history": 0.15,
                "brain_injury_history": 0.20,
                "psychiatric_diagnosis": 0.50,
                "substance_use": 0.30,
                "suicide_ideation": 0.20,
                "psychotropic_medication": 0.45,
                "birth_complications": 0.30,
                "extreme_poverty": 0.25,
                "education_access_issues": 0.35,
                "healthcare_access": 0.40,
                "disability_diagnosis": 0.25,
                "breastfed_infancy": 0.30,
                "violence_exposure": 0.40,
            },
            "very_high_risk": {
                "consanguinity": 0.30,
                "family_neuro_history": 0.70,
                "seizures_history": 0.35,
                "brain_injury_history": 0.40,
                "psychiatric_diagnosis": 0.80,
                "substance_use": 0.50,
                "suicide_ideation": 0.40,
                "psychotropic_medication": 0.75,
                "birth_complications": 0.50,
                "extreme_poverty": 0.45,
                "education_access_issues": 0.60,
                "healthcare_access": 0.15,
                "disability_diagnosis": 0.45,
                "breastfed_infancy": 0.10,
                "violence_exposure": 0.70,
            },
        }

        probs = risk_probabilities[risk_profile]

        # Generate binary features
        for feature in probs:
            # Add some age-related modifications
            prob = probs[feature]

            # Young age affects certain features
            if age < 18:
                if feature == "substance_use":
                    prob *= 0.3  # Less likely in children
                elif feature == "psychotropic_medication":
                    prob *= 0.7  # Slightly less likely
                elif feature == "birth_complications":
                    prob *= 1.2  # More relevant for young subjects

            # Elderly age affects features
            if age > 65:
                if feature == "substance_use":
                    prob *= 0.5
                elif feature == "disability_diagnosis":
                    prob *= 1.5

            # Gender effects (minimal, based on epidemiology)
            if gender == "M":
                if feature == "suicide_ideation":
                    prob *= 1.2
                elif feature == "substance_use":
                    prob *= 1.1

            # Ensure probability is valid
            prob = min(max(prob, 0.0), 1.0)

            # Generate boolean value
            features[feature] = bool(np.random.random() < prob)

        # Logical constraints
        # If on psychotropic medication, likely has psychiatric diagnosis
        if features["psychotropic_medication"] and not features["psychiatric_diagnosis"]:
            features["psychiatric_diagnosis"] = np.random.random() < 0.8

        # If suicide ideation, very likely has psychiatric diagnosis
        if features["suicide_ideation"] and not features["psychiatric_diagnosis"]:
            features["psychiatric_diagnosis"] = True

        # Social support level (correlated with other factors)
        if features["extreme_poverty"] or features["violence_exposure"]:
            support_probs = [0.6, 0.3, 0.1]  # More likely isolated
        elif features["healthcare_access"]:
            support_probs = [0.1, 0.3, 0.6]  # More likely supported
        else:
            support_probs = [0.2, 0.5, 0.3]  # Balanced

        features["social_support_level"] = np.random.choice(
            ["isolated", "moderate", "supported"], p=support_probs
        )

        return features

    def calculate_risk_score(self, features: Dict) -> float:
        """
        Calculate risk score based on features using weighted logic.

        Args:
            features: Feature dictionary

        Returns:
            Risk score between 0 and 1
        """
        score = 0.0

        # Get weights from feature definitions
        for feature_def in self.feature_defs["features"]:
            feature_name = feature_def["name"]
            weight = feature_def.get("weight", 0.1)

            if feature_name in ["age", "gender"]:
                continue  # Skip demographics in basic calculation

            if feature_name == "social_support_level":
                # Map to numeric
                mapping = {"isolated": 0, "moderate": 0.5, "supported": 1}
                value = mapping.get(features[feature_name], 0.5)
                if feature_def["risk_direction"] == "varies":
                    # Isolated increases risk
                    score += weight * (1 - value)
            else:
                # Binary features
                if features.get(feature_name, False):
                    if feature_def["risk_direction"] == "positive":
                        score += weight
                    elif feature_def["risk_direction"] == "negative":
                        score -= weight * 0.5  # Protective factors have less weight

        # Age modifier
        age = features["age"]
        if age < 5:
            score *= 1.2  # Very young age increases risk
        elif age > 65:
            score *= 1.1  # Elderly age slightly increases risk

        # Normalize to 0-1 range
        score = max(0.0, min(1.0, score))

        # Add some noise
        noise = np.random.normal(0, 0.05)
        score = max(0.0, min(1.0, score + noise))

        return score

    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate the complete synthetic dataset.

        Returns:
            DataFrame with all samples
        """
        logger.info(f"Generating {self.n_samples} synthetic samples...")

        data = []

        for i in range(self.n_samples):
            # Generate demographics
            age, gender = self.generate_demographics()

            # Assign risk profile
            risk_profile = self.generate_risk_profile()

            # Generate features
            features = self.generate_features(age, gender, risk_profile)

            # Calculate risk score
            risk_score = self.calculate_risk_score(features)

            # Determine risk level
            if risk_score < 0.3:
                risk_level = "low"
            elif risk_score < 0.7:
                risk_level = "moderate"
            else:
                risk_level = "high"

            # Create record
            record = {
                "id": i + 1,
                "assessment_date": datetime.now() - timedelta(days=random.randint(0, 365)),
                "risk_profile_truth": risk_profile,  # Ground truth for validation
                "risk_score": risk_score,
                "risk_level": risk_level,
                **features,
            }

            data.append(record)

            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{self.n_samples} samples")

        df = pd.DataFrame(data)

        logger.info(f"Dataset generation complete. Shape: {df.shape}")
        logger.info(f"Risk distribution: {df['risk_level'].value_counts().to_dict()}")

        return df

    def save_dataset(self, df: pd.DataFrame, filename: str = None):
        """
        Save dataset to CSV file.

        Args:
            df: DataFrame to save
            filename: Output filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"synthetic_data_{timestamp}.csv"

        output_path = settings.synthetic_data_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")

        # Also save a JSON version of summary statistics
        stats = {
            "generation_date": datetime.now().isoformat(),
            "n_samples": len(df),
            "random_seed": self.random_seed,
            "risk_distribution": df["risk_level"].value_counts().to_dict(),
            "age_stats": {
                "mean": float(df["age"].mean()),
                "std": float(df["age"].std()),
                "min": int(df["age"].min()),
                "max": int(df["age"].max()),
            },
            "gender_distribution": df["gender"].value_counts().to_dict(),
            "feature_prevalence": {},
        }

        # Calculate feature prevalence
        binary_features = [
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

        for feature in binary_features:
            stats["feature_prevalence"][feature] = float(df[feature].mean())

        stats_path = output_path.with_suffix(".json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics saved to {stats_path}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic neurodevelopmental assessment data"
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=settings.synthetic_samples,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=settings.random_seed,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output filename (default: auto-generated with timestamp)",
    )

    args = parser.parse_args()

    logger.info("Starting synthetic data generation")
    logger.info(f"Parameters: samples={args.samples}, seed={args.seed}")

    # Generate data
    generator = SyntheticDataGenerator(n_samples=args.samples, random_seed=args.seed)

    df = generator.generate_dataset()
    generator.save_dataset(df, args.output)

    logger.info("Data generation completed successfully")

    # Print summary
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"\nRisk distribution:")
    print(df["risk_level"].value_counts())
    print(f"\nAge range: {df['age'].min()} - {df['age'].max()} (mean: {df['age'].mean():.1f})")
    print(f"\nGender distribution:")
    print(df["gender"].value_counts())
    print("\nTop 5 most prevalent risk factors:")

    prevalence = {}
    for col in df.columns:
        if col in [
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
            "disability_diagnosis",
            "violence_exposure",
        ]:
            prevalence[col] = df[col].mean()

    for factor, prev in sorted(prevalence.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {factor}: {prev:.2%}")

    print("=" * 60)


if __name__ == "__main__":
    main()
