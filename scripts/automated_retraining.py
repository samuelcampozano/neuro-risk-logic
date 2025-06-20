#!/usr/bin/env python3
"""
Automated model retraining service.
Can be scheduled via cron or run as a background service.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import schedule
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np

from app.config import settings
from app.database import SessionLocal
from app.models.assessment import Assessment
from scripts.retrain_with_real_data import IncrementalModelTrainer
from scripts.evaluate_model import ModelEvaluator
from loguru import logger

# Configure logging
logger.add(
    "logs/automated_retraining.log",
    rotation="1 week",
    retention="1 month",
    level="INFO"
)


class AutomatedRetrainingService:
    """Automated model retraining and monitoring service."""
    
    def __init__(self):
        """Initialize the retraining service."""
        self.config = self.load_config()
        self.last_training = None
        self.training_lock = threading.Lock()
        self.metrics_history = []
        
    def load_config(self) -> Dict[str, Any]:
        """Load service configuration."""
        config_path = Path("config/retraining_config.json")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "min_samples_for_retraining": 50,
            "retraining_schedule": "weekly",  # daily, weekly, monthly
            "performance_degradation_threshold": 0.02,
            "email_notifications": False,
            "email_recipients": [],
            "monitoring_metrics": ["auc_roc", "f1_score"],
            "data_quality_checks": True,
            "backup_models": True,
            "max_model_versions": 10
        }
    
    def check_retraining_conditions(self) -> Dict[str, Any]:
        """
        Check if retraining conditions are met.
        
        Returns:
            Dictionary with check results
        """
        db = SessionLocal()
        try:
            # Count new assessments since last training
            if self.last_training:
                new_assessments = db.query(Assessment).filter(
                    Assessment.assessment_date > self.last_training,
                    Assessment.consent_given == True
                ).count()
            else:
                new_assessments = db.query(Assessment).filter(
                    Assessment.consent_given == True
                ).count()
            
            # Check data drift (simplified - could be more sophisticated)
            recent_assessments = db.query(Assessment).filter(
                Assessment.assessment_date >= datetime.now() - timedelta(days=30)
            ).all()
            
            if recent_assessments:
                recent_risk_mean = sum(a.risk_score for a in recent_assessments) / len(recent_assessments)
            else:
                recent_risk_mean = None
            
            # Check model performance degradation
            performance_degraded = self.check_performance_degradation()
            
            results = {
                "new_assessments": new_assessments,
                "sufficient_samples": new_assessments >= self.config["min_samples_for_retraining"],
                "recent_risk_mean": recent_risk_mean,
                "performance_degraded": performance_degraded,
                "should_retrain": (
                    new_assessments >= self.config["min_samples_for_retraining"] or
                    performance_degraded
                )
            }
            
            return results
            
        finally:
            db.close()
    
    def check_performance_degradation(self) -> bool:
        """
        Check if model performance has degraded.
        
        Returns:
            True if performance has degraded beyond threshold
        """
        if len(self.metrics_history) < 2:
            return False
        
        # Compare recent performance to baseline
        baseline_metrics = self.metrics_history[0]
        recent_metrics = self.metrics_history[-1]
        
        for metric in self.config["monitoring_metrics"]:
            if metric in baseline_metrics and metric in recent_metrics:
                degradation = baseline_metrics[metric] - recent_metrics[metric]
                if degradation > self.config["performance_degradation_threshold"]:
                    logger.warning(f"Performance degradation detected: {metric} dropped by {degradation:.3f}")
                    return True
        
        return False
    
    def run_retraining(self) -> Dict[str, Any]:
        """
        Execute the retraining pipeline.
        
        Returns:
            Retraining results
        """
        with self.training_lock:
            logger.info("="*60)
            logger.info("Starting automated retraining check")
            logger.info("="*60)
            
            # Check conditions
            conditions = self.check_retraining_conditions()
            logger.info(f"Retraining conditions: {conditions}")
            
            if not conditions["should_retrain"]:
                logger.info("Retraining conditions not met")
                return {
                    "status": "skipped",
                    "reason": "conditions_not_met",
                    "conditions": conditions
                }
            
            # Run retraining
            trainer = IncrementalModelTrainer(
                min_new_samples=self.config["min_samples_for_retraining"]
            )
            
            retrain_result = trainer.run_retraining_pipeline(force=False)
            
            # Update last training time
            if retrain_result["status"] == "success":
                self.last_training = datetime.now()
                
                # Evaluate new model
                evaluator = ModelEvaluator()
                evaluator.load_model()
                X_test, y_test = evaluator.load_test_data()
                metrics = evaluator.evaluate_predictions(X_test, y_test)
                
                # Add to metrics history
                self.metrics_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "version": evaluator.metadata.get("version", "unknown"),
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                })
                
                # Clean up old models if needed
                self.cleanup_old_models()
                
                # Send notification if configured
                if self.config["email_notifications"]:
                    self.send_notification(retrain_result, metrics)
            
            return retrain_result
    
    def cleanup_old_models(self):
        """Remove old model versions to save space."""
        model_files = sorted(settings.models_dir.glob("model_v*.pkl"))
        
        if len(model_files) > self.config["max_model_versions"]:
            # Keep only the most recent versions
            files_to_remove = model_files[:-self.config["max_model_versions"]]
            
            for file_path in files_to_remove:
                try:
                    # Remove model file and metadata
                    file_path.unlink()
                    metadata_path = file_path.with_suffix('.json')
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    logger.info(f"Removed old model: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {e}")
    
    def send_notification(self, retrain_result: Dict[str, Any], metrics: Dict[str, Any]):
        """
        Send email notification about retraining results.
        
        Args:
            retrain_result: Retraining results
            metrics: Model performance metrics
        """
        if not self.config["email_recipients"]:
            return
        
        try:
            # Create message
            subject = f"NeuroRiskLogic Model Retraining - {retrain_result['status'].upper()}"
            
            body = f"""
            Model Retraining Report
            ======================
            
            Status: {retrain_result['status']}
            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Performance Metrics:
            - Accuracy: {metrics.get('accuracy', 0):.3f}
            - Precision: {metrics.get('precision', 0):.3f}
            - Recall: {metrics.get('recall', 0):.3f}
            - F1-Score: {metrics.get('f1_score', 0):.3f}
            - AUC-ROC: {metrics.get('auc_roc', 0):.3f}
            
            Data Composition:
            {json.dumps(retrain_result.get('data_composition', {}), indent=2)}
            
            This is an automated notification from the NeuroRiskLogic system.
            """
            
            # Send email (simplified - would need SMTP configuration)
            logger.info(f"Notification would be sent to: {self.config['email_recipients']}")
            logger.info(f"Subject: {subject}")
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def monitor_data_quality(self) -> Dict[str, Any]:
        """
        Monitor data quality metrics.
        
        Returns:
            Data quality report
        """
        db = SessionLocal()
        try:
            # Get recent assessments
            recent_date = datetime.now() - timedelta(days=30)
            recent_assessments = db.query(Assessment).filter(
                Assessment.assessment_date >= recent_date
            ).all()
            
            if not recent_assessments:
                return {"status": "no_recent_data"}
            
            # Calculate quality metrics
            quality_metrics = {
                "total_recent_assessments": len(recent_assessments),
                "risk_score_distribution": {},
                "feature_completeness": {},
                "anomalies": []
            }
            
            # Risk score distribution
            risk_scores = [a.risk_score for a in recent_assessments]
            quality_metrics["risk_score_distribution"] = {
                "mean": sum(risk_scores) / len(risk_scores),
                "min": min(risk_scores),
                "max": max(risk_scores),
                "std": np.std(risk_scores) if len(risk_scores) > 1 else 0
            }
            
            # Check for anomalies
            for assessment in recent_assessments:
                # Example: Check for impossible combinations
                if assessment.age < 0 or assessment.age > 120:
                    quality_metrics["anomalies"].append(f"Invalid age: {assessment.age} (ID: {assessment.id})")
                
                if assessment.birth_complications and assessment.age > 50:
                    quality_metrics["anomalies"].append(
                        f"Birth complications reported for older subject (ID: {assessment.id})"
                    )
            
            return quality_metrics
            
        finally:
            db.close()
    
    def run_scheduled_job(self):
        """Run the scheduled retraining job."""
        try:
            logger.info("Running scheduled retraining job")
            
            # Monitor data quality first
            if self.config["data_quality_checks"]:
                quality_report = self.monitor_data_quality()
                logger.info(f"Data quality report: {quality_report}")
                
                if quality_report.get("anomalies"):
                    logger.warning(f"Data quality issues detected: {len(quality_report['anomalies'])} anomalies")
            
            # Run retraining
            result = self.run_retraining()
            
            # Save service state
            self.save_state()
            
            logger.info(f"Scheduled job completed with status: {result['status']}")
            
        except Exception as e:
            logger.error(f"Error in scheduled job: {e}")
    
    def save_state(self):
        """Save service state for persistence."""
        state = {
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "metrics_history": self.metrics_history[-50:],  # Keep last 50 entries
            "last_run": datetime.now().isoformat()
        }
        
        state_path = Path("data/retraining_state.json")
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        """Load service state from disk."""
        state_path = Path("data/retraining_state.json")
        
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            if state.get("last_training"):
                self.last_training = datetime.fromisoformat(state["last_training"])
            
            self.metrics_history = state.get("metrics_history", [])
            
            logger.info(f"Loaded state from {state_path}")
    
    def start_service(self):
        """Start the automated retraining service."""
        logger.info("Starting Automated Retraining Service")
        
        # Load previous state
        self.load_state()
        
        # Schedule jobs based on configuration
        schedule_type = self.config["retraining_schedule"]
        
        if schedule_type == "daily":
            schedule.every().day.at("02:00").do(self.run_scheduled_job)
            logger.info("Scheduled daily retraining at 02:00")
        elif schedule_type == "weekly":
            schedule.every().monday.at("03:00").do(self.run_scheduled_job)
            logger.info("Scheduled weekly retraining on Mondays at 03:00")
        elif schedule_type == "monthly":
            schedule.every(30).days.do(self.run_scheduled_job)
            logger.info("Scheduled monthly retraining")
        
        # Also schedule data quality monitoring
        schedule.every(6).hours.do(self.monitor_data_quality)
        
        # Run initial check
        self.run_scheduled_job()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automated model retraining service"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (don't start service)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check conditions without retraining"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Initialize service
    service = AutomatedRetrainingService()
    
    if args.config:
        # Load custom configuration
        with open(args.config, 'r') as f:
            service.config = json.load(f)
    
    if args.check_only:
        # Just check conditions
        conditions = service.check_retraining_conditions()
        print("\nRetraining Conditions Check:")
        print("="*40)
        for key, value in conditions.items():
            print(f"{key}: {value}")
        print("="*40)
        
    elif args.once:
        # Run once
        result = service.run_retraining()
        print(f"\nRetraining completed with status: {result['status']}")
        
    else:
        # Start service
        try:
            service.start_service()
        except KeyboardInterrupt:
            logger.info("Service stopped by user")


if __name__ == "__main__":
    main()