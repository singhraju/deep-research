"""
Simple S3 Checkpoint Manager for Auto-Resume
Works in conjunction with job_tracker.py for detailed history
"""

import json
import boto3
from datetime import datetime
from typing import Dict, Optional, Tuple


class CheckpointManager:
    """
    Manages simple S3 checkpoint file for auto-resume functionality
    
    Checkpoint file contains only:
    - run_id: Current/last run identifier
    - status: running, success, failed
    - environment: dev, uat, prod
    - last_updated: timestamp
    
    Detailed tracking is handled by job_tracker.py in Snowflake
    """
    
    def __init__(self, s3_client: boto3.client, bucket: str, env: str):
        """
        Initialize Checkpoint Manager
        
        Args:
            s3_client: Boto3 S3 client
            bucket: S3 bucket name
            env: Environment (dv, ts, pl, pr)
        """
        self.s3_client = s3_client
        self.bucket = bucket
        self.env = env
        self.checkpoint_key = "deep_rsrch/dataz/outbound/checkpoint.json"
    
    def load_checkpoint(self) -> Optional[Dict]:
        """
        Load checkpoint from S3
        
        Returns:
            Dict with checkpoint data, or None if not exists
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=self.checkpoint_key
            )
            checkpoint = json.loads(response['Body'].read())
            print(f"📋 Loaded checkpoint from S3: {checkpoint['run_id']} ({checkpoint['status']})")
            return checkpoint
        except self.s3_client.exceptions.NoSuchKey:
            print("📋 No checkpoint found in S3 - starting fresh")
            return None
        except Exception as e:
            print(f"⚠️  Warning: Could not load checkpoint: {str(e)}")
            return None
    
    def should_resume(self) -> Tuple[bool, Optional[str]]:
        """
        Check if pipeline should resume from previous failed run
        
        Returns:
            Tuple of (should_resume: bool, run_id: str or None)
        """
        checkpoint = self.load_checkpoint()
        
        if checkpoint is None:
            return False, None
        
        status = checkpoint.get('status')
        run_id = checkpoint.get('run_id')
        
        if status == 'failed':
            print(f"🔄 Detected failed run: {run_id}")
            print(f"   Will resume automatically...")
            return True, run_id
        elif status == 'running':
            print(f"⚠️  Previous run {run_id} was interrupted (status: running)")
            print(f"   Will resume automatically...")
            return True, run_id
        elif status == 'success':
            print(f"✅ Previous run {run_id} completed successfully")
            print(f"   Starting new run...")
            return False, None
        else:
            print(f"⚠️  Unknown checkpoint status: {status}")
            return False, None
    
    def create_checkpoint(self, run_id: str, status: str = "running") -> Dict:
        """
        Create new checkpoint file
        
        Args:
            run_id: Run identifier
            status: Initial status (default: running)
            
        Returns:
            Dict with checkpoint data
        """
        checkpoint = {
            "run_id": run_id,
            "status": status,
            "environment": self.env,
            "last_updated": datetime.now().isoformat()
        }
        
        self.save_checkpoint(checkpoint)
        print(f"📋 Created checkpoint: {run_id} (status: {status})")
        return checkpoint
    
    def update_status(self, run_id: str, status: str):
        """
        Update checkpoint status
        
        Args:
            run_id: Run identifier
            status: New status (running, success, failed)
        """
        checkpoint = self.load_checkpoint()
        
        if checkpoint is None:
            # Create new checkpoint if doesn't exist
            checkpoint = {
                "run_id": run_id,
                "environment": self.env
            }
        
        checkpoint["status"] = status
        checkpoint["last_updated"] = datetime.now().isoformat()
        
        self.save_checkpoint(checkpoint)
        print(f"📋 Updated checkpoint: {run_id} → {status}")
    
    def save_checkpoint(self, checkpoint: Dict):
        """
        Save checkpoint to S3
        
        Args:
            checkpoint: Checkpoint dictionary
        """
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=self.checkpoint_key,
                Body=json.dumps(checkpoint, indent=2),
                ContentType='application/json'
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not save checkpoint: {str(e)}")
    
    def mark_success(self, run_id: str):
        """
        Mark run as successful
        
        Args:
            run_id: Run identifier
        """
        self.update_status(run_id, "success")
        print(f"✅ Run {run_id} marked as SUCCESS")
    
    def mark_failed(self, run_id: str):
        """
        Mark run as failed
        
        Args:
            run_id: Run identifier
        """
        self.update_status(run_id, "failed")
        print(f"❌ Run {run_id} marked as FAILED")
    
    def get_current_run_id(self) -> Optional[str]:
        """
        Get current run ID from checkpoint
        
        Returns:
            Run ID string, or None if no checkpoint
        """
        checkpoint = self.load_checkpoint()
        return checkpoint.get('run_id') if checkpoint else None
    
    def get_checkpoint_info(self) -> Dict:
        """
        Get checkpoint information for display
        
        Returns:
            Dict with checkpoint details
        """
        checkpoint = self.load_checkpoint()
        
        if checkpoint is None:
            return {
                "exists": False,
                "message": "No checkpoint found - will start fresh run"
            }
        
        status = checkpoint.get('status')
        run_id = checkpoint.get('run_id')
        last_updated = checkpoint.get('last_updated', 'Unknown')
        
        if status == 'failed':
            message = f"Failed run detected: {run_id} (will auto-resume)"
        elif status == 'running':
            message = f"Interrupted run detected: {run_id} (will auto-resume)"
        elif status == 'success':
            message = f"Previous run successful: {run_id} (will start new run)"
        else:
            message = f"Unknown status: {status}"
        
        return {
            "exists": True,
            "run_id": run_id,
            "status": status,
            "last_updated": last_updated,
            "message": message
        }


def generate_run_id() -> str:
    """
    Generate unique run ID based on timestamp
    
    Returns:
        Run ID string (format: YYYYMMDD_HHMM)
    """
    return datetime.now().strftime("%Y%m%d_%H%M")
