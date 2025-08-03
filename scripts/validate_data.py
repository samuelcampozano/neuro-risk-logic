#!/usr/bin/env python3
"""
Validate real clinical data before preprocessing.
Checks for data quality issues and provides recommendations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


def validate_clinical_data(file_path: str) -> Dict:
    """
    Validate clinical data file.

    Returns:
        Validation report with issues and recommendations
    """
    df = (
        pd.read_excel(file_path)
        if file_path.endswith((".xlsx", ".xls"))
        else pd.read_csv(file_path)
    )

    report = {
        "file_info": {
            "path": file_path,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
        },
        "issues": [],
        "warnings": [],
        "recommendations": [],
        "column_analysis": {},
    }

    # Check for minimum rows
    if len(df) < 50:
        report["issues"].append(f"Too few samples ({len(df)}). Minimum 50 required for training.")

    # Analyze each column
    for col in df.columns:
        col_analysis = {
            "dtype": str(df[col].dtype),
            "missing": df[col].isna().sum(),
            "missing_pct": f"{(df[col].isna().sum() / len(df)) * 100:.1f}%",
            "unique_values": df[col].nunique(),
        }

        # Check for high missing data
        if df[col].isna().sum() / len(df) > 0.5:
            report["warnings"].append(f"Column '{col}' has >50% missing data")

        # Check for constant columns
        if df[col].nunique() == 1:
            report["warnings"].append(f"Column '{col}' has only one unique value")

        # Sample values for inspection
        col_analysis["sample_values"] = df[col].dropna().head(5).tolist()

        report["column_analysis"][col] = col_analysis

    # Check for required features
    expected_features = [
        "edad",
        "sexo",
        "consanguinidad",
        "antecedentes",
        "convulsiones",
        "medicacion",
        "apoyo_social",
    ]

    found_features = []
    for expected in expected_features:
        found = False
        for col in df.columns:
            if expected.lower() in col.lower():
                found_features.append(col)
                found = True
                break
        if not found:
            report["warnings"].append(f"Expected feature '{expected}' not found")

    # Generate recommendations
    if len(df) < 200:
        report["recommendations"].append(
            "Consider collecting more data. Current dataset may be too small for robust training."
        )

    if len(report["warnings"]) > 5:
        report["recommendations"].append(
            "Multiple data quality issues detected. Review and clean data before training."
        )

    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validate_data.py <data_file>")
        sys.exit(1)

    report = validate_clinical_data(sys.argv[1])
    print(json.dumps(report, indent=2, ensure_ascii=False))
