#!/usr/bin/env python3
"""
Preprocess real clinical data for NeuroRiskLogic ML training.
Maps columns from the actual dataset to our expected features.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")

from app.config import settings
from loguru import logger

# Configure logging
logger.add("logs/data_preprocessing.log", rotation="10 MB", level="INFO")


class RealDataPreprocessor:
    """Preprocess real clinical data for ML training."""

    def __init__(self, input_file: str):
        """
        Initialize preprocessor.

        Args:
            input_file: Path to input CSV/Excel file
        """
        self.input_file = Path(input_file)
        self.df = None
        self.processed_df = None

        # Define column mappings from your dataset to our features
        self.column_mappings = {
            # Demographics
            "age": "Edad",  # Age column
            "gender": "Sexo",  # Gender column
            # Clinical-Genetic Features
            "consanguinity": "consanguinidad",  # Parents blood-related
            "family_neuro_history": "trastorno del neurodesarrollo",  # Family history
            "seizures_history": "convulsiones",  # Seizures
            "brain_injury_history": "traumatismo_craneal",  # Brain injury
            "psychiatric_diagnosis": "diagnostico_psiquiatrico",  # Psychiatric diagnosis
            "substance_use": "consumo_sustancias",  # Substance use
            "suicide_ideation": "ideacion_suicida",  # Suicide ideation
            "psychotropic_medication": "medicacion_psicotropica",  # Medication
            # Sociodemographic Features
            "birth_complications": "complicaciones_parto",  # Birth complications
            "extreme_poverty": "pobreza_extrema",  # Extreme poverty
            "education_access_issues": "acceso_educacion",  # Education access
            "healthcare_access": "acceso_salud",  # Healthcare access
            "disability_diagnosis": "discapacidad",  # Disability
            "breastfed_infancy": "lactancia_materna",  # Breastfeeding
            "violence_exposure": "violencia",  # Violence exposure
            "social_support_level": "apoyo_social",  # Social support
        }

        # Features that might need special handling
        self.binary_features = [
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

        logger.info(f"Initialized preprocessor for {input_file}")

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV or Excel file."""
        try:
            if self.input_file.suffix.lower() in [".xlsx", ".xls"]:
                self.df = pd.read_excel(self.input_file)
            else:
                self.df = pd.read_csv(self.input_file, encoding="utf-8")

            logger.info(f"Loaded {len(self.df)} rows from {self.input_file}")
            logger.info(f"Columns found: {list(self.df.columns)}")

            return self.df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def analyze_columns(self) -> Dict[str, List[str]]:
        """Analyze which columns are present and missing."""
        if self.df is None:
            self.load_data()

        analysis = {
            "found_mappings": {},
            "missing_features": [],
            "unmapped_columns": [],
            "data_types": {},
        }

        # Check each expected feature
        for feature, expected_col in self.column_mappings.items():
            # Try different variations of column names
            found = False
            for col in self.df.columns:
                if expected_col.lower() in col.lower() or col.lower() in expected_col.lower():
                    analysis["found_mappings"][feature] = col
                    analysis["data_types"][feature] = str(self.df[col].dtype)
                    found = True
                    break

            if not found:
                analysis["missing_features"].append(feature)

        # Find unmapped columns
        mapped_cols = list(analysis["found_mappings"].values())
        analysis["unmapped_columns"] = [col for col in self.df.columns if col not in mapped_cols]

        return analysis

    def clean_binary_column(self, series: pd.Series) -> pd.Series:
        """
        Clean and convert a column to binary (0/1).

        Handles various representations like:
        - Yes/No, Si/No, Y/N, S/N
        - True/False, 1/0
        - Present/Absent, Presente/Ausente
        """
        if series.dtype in ["int64", "float64"]:
            # Already numeric, just ensure 0/1
            return (series > 0).astype(int)

        # Convert to string and lowercase
        series_str = series.astype(str).str.lower().str.strip()

        # Define positive values
        positive_values = [
            "yes",
            "si",
            "sí",
            "y",
            "s",
            "1",
            "true",
            "verdadero",
            "present",
            "presente",
            "positive",
            "positivo",
            "x",
        ]

        # Convert to binary
        return series_str.isin(positive_values).astype(int)

    def clean_gender_column(self, series: pd.Series) -> pd.Series:
        """Clean and standardize gender column."""
        gender_map = {
            "m": "M",
            "masculino": "M",
            "male": "M",
            "hombre": "M",
            "h": "M",
            "f": "F",
            "femenino": "F",
            "female": "F",
            "mujer": "F",
            "o": "Other",
            "otro": "Other",
            "other": "Other",
        }

        series_clean = series.astype(str).str.lower().str.strip()
        return series_clean.map(gender_map).fillna("Other")

    def clean_social_support(self, series: pd.Series) -> pd.Series:
        """Clean and standardize social support level."""
        # Try to infer from the data
        if series.dtype in ["int64", "float64"]:
            # If numeric, map to categories
            tertiles = series.quantile([0.33, 0.67])
            conditions = [
                series <= tertiles[0.33],
                (series > tertiles[0.33]) & (series <= tertiles[0.67]),
                series > tertiles[0.67],
            ]
            choices = ["isolated", "moderate", "supported"]
            return pd.Series(np.select(conditions, choices, default="moderate"))

        # If text, try to map
        support_map = {
            "bajo": "isolated",
            "low": "isolated",
            "aislado": "isolated",
            "medio": "moderate",
            "moderate": "moderate",
            "moderado": "moderate",
            "alto": "supported",
            "high": "supported",
            "apoyado": "supported",
        }

        series_clean = series.astype(str).str.lower().str.strip()
        return series_clean.map(support_map).fillna("moderate")

    def preprocess_data(self) -> pd.DataFrame:
        """
        Main preprocessing pipeline.

        Returns:
            Preprocessed DataFrame ready for ML
        """
        if self.df is None:
            self.load_data()

        logger.info("Starting data preprocessing...")

        # Get column analysis
        analysis = self.analyze_columns()
        logger.info(f"Found {len(analysis['found_mappings'])} mapped features")
        logger.info(f"Missing features: {analysis['missing_features']}")

        # Create processed dataframe with our expected columns
        processed_data = {}

        # Process each feature
        for feature, source_col in analysis["found_mappings"].items():
            logger.info(f"Processing {feature} from column {source_col}")

            if feature == "age":
                # Clean age - remove outliers, handle missing
                age_series = pd.to_numeric(self.df[source_col], errors="coerce")
                age_series = age_series.clip(lower=0, upper=120)
                processed_data["age"] = age_series.fillna(age_series.median())

            elif feature == "gender":
                processed_data["gender"] = self.clean_gender_column(self.df[source_col])

            elif feature == "social_support_level":
                processed_data["social_support_level"] = self.clean_social_support(
                    self.df[source_col]
                )

            elif feature in self.binary_features:
                processed_data[feature] = self.clean_binary_column(self.df[source_col])

            else:
                # Default handling
                processed_data[feature] = self.df[source_col]

        # Handle missing features with defaults
        for missing_feature in analysis["missing_features"]:
            logger.warning(f"Feature {missing_feature} not found, using defaults")

            if missing_feature in self.binary_features:
                # Use median risk assumption (0.3 probability)
                processed_data[missing_feature] = np.random.binomial(1, 0.3, len(self.df))
            elif missing_feature == "social_support_level":
                processed_data[missing_feature] = "moderate"
            elif missing_feature == "age":
                processed_data[missing_feature] = 30  # Default age
            elif missing_feature == "gender":
                processed_data[missing_feature] = "Other"

        # Create final dataframe
        self.processed_df = pd.DataFrame(processed_data)

        # Add metadata columns
        self.processed_df["source_file"] = self.input_file.name
        self.processed_df["processing_date"] = datetime.now()

        # Calculate risk score for pseudo-labeling
        self.processed_df["risk_score"] = self.calculate_risk_scores()
        self.processed_df["risk_level"] = pd.cut(
            self.processed_df["risk_score"],
            bins=[0, 0.3, 0.7, 1.0],
            labels=["low", "moderate", "high"],
        )

        logger.info(f"Preprocessing complete. Shape: {self.processed_df.shape}")
        logger.info(f"Risk distribution:\n{self.processed_df['risk_level'].value_counts()}")

        return self.processed_df

    def calculate_risk_scores(self) -> pd.Series:
        """
        Calculate risk scores based on feature weights.
        This creates pseudo-labels for training.
        """
        # Load feature weights
        with open(settings.get_feature_definitions_path(), "r") as f:
            feature_defs = json.load(f)

        risk_scores = pd.Series(0.0, index=self.processed_df.index)

        for feature_def in feature_defs["features"]:
            feature_name = feature_def["name"]
            weight = feature_def.get("weight", 0.1)
            risk_direction = feature_def.get("risk_direction", "positive")

            if feature_name in self.processed_df.columns:
                if feature_name in self.binary_features:
                    # Binary features
                    if risk_direction == "positive":
                        risk_scores += self.processed_df[feature_name] * weight
                    else:  # negative (protective)
                        risk_scores -= self.processed_df[feature_name] * weight * 0.5

                elif feature_name == "social_support_level":
                    # Categorical feature
                    support_scores = {"isolated": weight, "moderate": weight * 0.5, "supported": 0}
                    risk_scores += (
                        self.processed_df[feature_name].map(support_scores).fillna(weight * 0.5)
                    )

        # Normalize to 0-1 range
        risk_scores = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min())

        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.05, len(risk_scores))
        risk_scores = (risk_scores + noise).clip(0, 1)

        return risk_scores

    def save_processed_data(self, output_path: Optional[str] = None):
        """Save processed data to CSV."""
        if self.processed_df is None:
            raise ValueError("No processed data to save. Run preprocess_data() first.")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = settings.synthetic_data_dir / f"processed_real_data_{timestamp}.csv"

        self.processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

        # Save preprocessing report
        report = {
            "processing_date": datetime.now().isoformat(),
            "input_file": str(self.input_file),
            "input_rows": len(self.df),
            "output_rows": len(self.processed_df),
            "column_analysis": self.analyze_columns(),
            "risk_distribution": self.processed_df["risk_level"].value_counts().to_dict(),
            "missing_data_summary": {
                col: self.processed_df[col].isna().sum() for col in self.processed_df.columns
            },
        }

        report_path = Path(output_path).with_suffix(".json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Saved preprocessing report to {report_path}")

        return output_path

    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate data quality report."""
        if self.processed_df is None:
            raise ValueError("No processed data. Run preprocess_data() first.")

        report = {
            "total_records": len(self.processed_df),
            "features": {},
            "correlations": {},
            "recommendations": [],
        }

        # Analyze each feature
        for col in self.processed_df.columns:
            if col in ["source_file", "processing_date"]:
                continue

            feature_report = {
                "type": str(self.processed_df[col].dtype),
                "missing": self.processed_df[col].isna().sum(),
                "missing_pct": (
                    f"{(self.processed_df[col].isna().sum() / len(self.processed_df)) * 100:.1f}%"
                ),
            }

            if col in self.binary_features:
                feature_report["distribution"] = self.processed_df[col].value_counts().to_dict()
                feature_report["positive_rate"] = f"{self.processed_df[col].mean() * 100:.1f}%"

            elif col == "age":
                feature_report["stats"] = {
                    "mean": self.processed_df[col].mean(),
                    "std": self.processed_df[col].std(),
                    "min": self.processed_df[col].min(),
                    "max": self.processed_df[col].max(),
                }

            report["features"][col] = feature_report

        # Check correlations with risk score
        risk_correlations = {}
        for col in self.binary_features:
            if col in self.processed_df.columns:
                corr = self.processed_df[col].corr(self.processed_df["risk_score"])
                risk_correlations[col] = round(corr, 3)

        report["correlations"] = dict(
            sorted(risk_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        # Generate recommendations
        missing_features = self.analyze_columns()["missing_features"]
        if missing_features:
            report["recommendations"].append(
                f"Missing features: {', '.join(missing_features)}. Consider collecting this data."
            )

        low_variance_features = [
            col
            for col in self.binary_features
            if col in self.processed_df.columns
            and (self.processed_df[col].mean() < 0.05 or self.processed_df[col].mean() > 0.95)
        ]

        if low_variance_features:
            features_str = ", ".join(low_variance_features)
            report["recommendations"].append(
                f"Low variance features: {features_str}. May not be informative."
            )

        return report


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess real clinical data for NeuroRiskLogic")
    parser.add_argument("input_file", type=str, help="Path to input CSV or Excel file")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output file path (default: auto-generated)"
    )
    parser.add_argument("--report", action="store_true", help="Generate detailed quality report")

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        return

    # Process data
    preprocessor = RealDataPreprocessor(args.input_file)

    # Analyze columns first
    print("\n" + "=" * 60)
    print("COLUMN ANALYSIS")
    print("=" * 60)

    analysis = preprocessor.analyze_columns()
    print(f"\nFound mappings ({len(analysis['found_mappings'])}):")
    for feature, col in analysis["found_mappings"].items():
        print(f"  {feature:25} -> {col}")

    if analysis["missing_features"]:
        print(f"\nMissing features ({len(analysis['missing_features'])}):")
        for feature in analysis["missing_features"]:
            print(f"  - {feature}")

    print("\n" + "=" * 60)

    # Preprocess data
    processed_df = preprocessor.preprocess_data()

    # Save processed data
    output_path = preprocessor.save_processed_data(args.output)

    # Generate quality report
    if args.report:
        report = preprocessor.generate_quality_report()
        print("\n" + "=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        print(f"Total records: {report['total_records']}")
        print(f"\nTop risk correlations:")
        for feature, corr in list(report["correlations"].items())[:10]:
            print(f"  {feature:25} : {corr:+.3f}")

        if report["recommendations"]:
            print(f"\nRecommendations:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")

    print("\n" + "=" * 60)
    print(f"✅ Processing complete!")
    print(f"Output saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
