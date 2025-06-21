#!/usr/bin/env python3
"""
Development tools for testing and debugging the NeuroRiskLogic system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import requests
from typing import Dict, Any, List

from app.config import settings
from app.database import SessionLocal, init_db
from app.models.assessment import Assessment
from app.auth import get_admin_token
from loguru import logger


class DevTools:
    """Development and testing utilities."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize dev tools."""
        self.base_url = base_url
        self.token = None
        
    def get_auth_token(self) -> str:
        """Get admin authentication token."""
        if not self.token:
            token_obj = get_admin_token()
            self.token = token_obj.access_token
        return self.token
    
    def simulate_assessments(self, count: int = 100, days: int = 30) -> List[Dict]:
        """
        Simulate assessment submissions over time.
        
        Args:
            count: Number of assessments to create
            days: Spread over this many days
            
        Returns:
            List of created assessments
        """
        logger.info(f"Simulating {count} assessments over {days} days")
        
        headers = {
            "Authorization": f"Bearer {self.get_auth_token()}"
        }
        
        results = []
        
        for i in range(count):
            # Random date within range
            date_offset = np.random.randint(0, days * 24 * 60)  # minutes
            assessment_date = datetime.now() - timedelta(minutes=date_offset)
            
            # Generate random but realistic data
            data = {
                "age": int(np.random.normal(35, 15)),
                "gender": np.random.choice(["M", "F", "Other"], p=[0.5, 0.48, 0.02]),
                "consanguinity": bool(np.random.random() < 0.05),
                "family_neuro_history": bool(np.random.random() < 0.2),
                "seizures_history": bool(np.random.random() < 0.08),
                "brain_injury_history": bool(np.random.random() < 0.1),
                "psychiatric_diagnosis": bool(np.random.random() < 0.25),
                "substance_use": bool(np.random.random() < 0.15),
                "suicide_ideation": bool(np.random.random() < 0.05),
                "psychotropic_medication": bool(np.random.random() < 0.2),
                "birth_complications": bool(np.random.random() < 0.12),
                "extreme_poverty": bool(np.random.random() < 0.1),
                "education_access_issues": bool(np.random.random() < 0.15),
                "healthcare_access": bool(np.random.random() < 0.7),
                "disability_diagnosis": bool(np.random.random() < 0.1),
                "social_support_level": np.random.choice(["isolated", "moderate", "supported"], 
                                                       p=[0.2, 0.5, 0.3]),
                "breastfed_infancy": bool(np.random.random() < 0.6),
                "violence_exposure": bool(np.random.random() < 0.2),
                "consent_given": True,
                "clinician_id": f"SIM_{i:04d}"
            }
            
            # Ensure age is valid
            data["age"] = max(0, min(120, data["age"]))
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/assessments",
                    json=data,
                    headers=headers
                )
                
                if response.status_code == 201:
                    result = response.json()
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Created {i + 1}/{count} assessments")
                else:
                    logger.error(f"Failed to create assessment: {response.text}")
                    
            except Exception as e:
                logger.error(f"Error creating assessment: {e}")
        
        logger.info(f"Successfully created {len(results)} assessments")
        return results
    
    def test_prediction_endpoint(self) -> Dict[str, Any]:
        """Test the prediction endpoint."""
        test_data = {
            "age": 30,
            "gender": "M",
            "consanguinity": False,
            "family_neuro_history": True,
            "seizures_history": False,
            "brain_injury_history": False,
            "psychiatric_diagnosis": True,
            "substance_use": False,
            "suicide_ideation": False,
            "psychotropic_medication": True,
            "birth_complications": False,
            "extreme_poverty": False,
            "education_access_issues": False,
            "healthcare_access": True,
            "disability_diagnosis": False,
            "social_support_level": "moderate",
            "breastfed_infancy": True,
            "violence_exposure": False
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/predict",
            json=test_data
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("Prediction endpoint test successful")
            logger.info(f"Risk level: {result['risk_level']}")
            logger.info(f"Risk score: {result['risk_score']:.3f}")
            return result
        else:
            logger.error(f"Prediction test failed: {response.text}")
            return {}
    
    def test_retraining_endpoint(self) -> Dict[str, Any]:
        """Test the retraining endpoint."""
        headers = {
            "Authorization": f"Bearer {self.get_auth_token()}"
        }
        
        data = {
            "force": False,
            "min_samples": 10,
            "include_synthetic": True
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/retrain/start",
            json=data,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Retraining started: {result['task_id']}")
            return result
        else:
            logger.error(f"Retraining test failed: {response.text}")
            return {}
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        health_status = {
            "api": False,
            "database": False,
            "model": False,
            "authentication": False
        }
        
        # Check API
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                health_status["api"] = health_data.get("status") == "healthy"
                health_status["database"] = health_data.get("services", {}).get("database", {}).get("status") == "connected"
                health_status["model"] = health_data.get("services", {}).get("ml_model", {}).get("status") == "loaded"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        # Check authentication
        try:
            token = self.get_auth_token()
            if token:
                health_status["authentication"] = True
        except:
            pass
        
        return health_status
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old assessment data."""
        db = SessionLocal()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Count assessments to delete
            count = db.query(Assessment).filter(
                Assessment.assessment_date < cutoff_date
            ).count()
            
            if count > 0:
                confirm = input(f"Delete {count} assessments older than {days} days? (y/N): ")
                if confirm.lower() == 'y':
                    db.query(Assessment).filter(
                        Assessment.assessment_date < cutoff_date
                    ).delete()
                    db.commit()
                    logger.info(f"Deleted {count} old assessments")
                else:
                    logger.info("Cleanup cancelled")
            else:
                logger.info("No old assessments to clean up")
                
        finally:
            db.close()
    
    def export_data(self, output_file: str = None):
        """Export all assessment data to CSV."""
        db = SessionLocal()
        try:
            assessments = db.query(Assessment).all()
            
            if not assessments:
                logger.warning("No assessments to export")
                return
            
            # Convert to DataFrame
            data = []
            for a in assessments:
                data.append({
                    'id': a.id,
                    'assessment_date': a.assessment_date,
                    'age': a.age,
                    'gender': a.gender,
                    'consanguinity': a.consanguinity,
                    'family_neuro_history': a.family_neuro_history,
                    'seizures_history': a.seizures_history,
                    'brain_injury_history': a.brain_injury_history,
                    'psychiatric_diagnosis': a.psychiatric_diagnosis,
                    'substance_use': a.substance_use,
                    'suicide_ideation': a.suicide_ideation,
                    'psychotropic_medication': a.psychotropic_medication,
                    'birth_complications': a.birth_complications,
                    'extreme_poverty': a.extreme_poverty,
                    'education_access_issues': a.education_access_issues,
                    'healthcare_access': a.healthcare_access,
                    'disability_diagnosis': a.disability_diagnosis,
                    'social_support_level': a.social_support_level,
                    'breastfed_infancy': a.breastfed_infancy,
                    'violence_exposure': a.violence_exposure,
                    'risk_score': a.risk_score,
                    'risk_level': a.risk_level,
                    'confidence_score': a.confidence_score,
                    'model_version': a.model_version,
                    'clinician_id': a.clinician_id
                })
            
            df = pd.DataFrame(data)
            
            if output_file is None:
                output_file = f"assessment_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(df)} assessments to {output_file}")
            
        finally:
            db.close()
    
    def reset_database(self):
        """Reset database (WARNING: Deletes all data)."""
        confirm = input("WARNING: This will delete ALL data. Are you sure? (type 'yes' to confirm): ")
        
        if confirm.lower() == 'yes':
            db = SessionLocal()
            try:
                # Delete all assessments
                count = db.query(Assessment).count()
                db.query(Assessment).delete()
                db.commit()
                logger.info(f"Deleted {count} assessments")
                
                # Reinitialize tables
                init_db()
                logger.info("Database reset complete")
                
            finally:
                db.close()
        else:
            logger.info("Database reset cancelled")
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_health": self.check_system_health(),
            "endpoints_tested": {},
            "database_stats": {},
            "model_info": {}
        }
        
        # Test endpoints
        logger.info("Testing prediction endpoint...")
        pred_result = self.test_prediction_endpoint()
        report["endpoints_tested"]["prediction"] = bool(pred_result)
        
        # Database stats
        db = SessionLocal()
        try:
            total_assessments = db.query(Assessment).count()
            recent_assessments = db.query(Assessment).filter(
                Assessment.assessment_date >= datetime.now() - timedelta(days=30)
            ).count()
            
            report["database_stats"] = {
                "total_assessments": total_assessments,
                "recent_assessments": recent_assessments
            }
        finally:
            db.close()
        
        # Model info
        try:
            response = requests.get(f"{self.base_url}/api/v1/model/info")
            if response.status_code == 200:
                report["model_info"] = response.json()
        except:
            pass
        
        # Save report
        report_path = Path("test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("SYSTEM TEST REPORT")
        print("="*60)
        print(f"Generated: {report['timestamp']}")
        print("\nSystem Health:")
        for component, status in report["system_health"].items():
            status_text = "✅ OK" if status else "❌ FAILED"
            print(f"  {component}: {status_text}")
        print("\nDatabase Statistics:")
        for stat, value in report["database_stats"].items():
            print(f"  {stat}: {value}")
        print("="*60)


def main():
    """Main CLI for development tools."""
    parser = argparse.ArgumentParser(
        description="NeuroRiskLogic development tools"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Simulate command
    simulate_parser = subparsers.add_parser('simulate', help='Simulate assessments')
    simulate_parser.add_argument('-n', '--count', type=int, default=100, help='Number of assessments')
    simulate_parser.add_argument('-d', '--days', type=int, default=30, help='Spread over days')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run system tests')
    test_parser.add_argument('--endpoint', choices=['predict', 'retrain', 'all'], default='all')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check system health')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export data')
    export_parser.add_argument('-o', '--output', help='Output file')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean old data')
    cleanup_parser.add_argument('-d', '--days', type=int, default=90, help='Days to keep')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset database')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate test report')
    
    args = parser.parse_args()
    
    # Initialize tools
    tools = DevTools()
    
    if args.command == 'simulate':
        tools.simulate_assessments(args.count, args.days)
    
    elif args.command == 'test':
        if args.endpoint in ['predict', 'all']:
            tools.test_prediction_endpoint()
        if args.endpoint in ['retrain', 'all']:
            tools.test_retraining_endpoint()
    
    elif args.command == 'health':
        health = tools.check_system_health()
        for component, status in health.items():
            status_text = "✅ OK" if status else "❌ FAILED"
            print(f"{component}: {status_text}")
    
    elif args.command == 'export':
        tools.export_data(args.output)
    
    elif args.command == 'cleanup':
        tools.cleanup_old_data(args.days)
    
    elif args.command == 'reset':
        tools.reset_database()
    
    elif args.command == 'report':
        tools.generate_test_report()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()