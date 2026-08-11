"""
S3 utility functions for saving agent payloads and results
"""

import boto3
import json
import os
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
from utils.config_loader import get_config_loader


class S3Logger:
    """
    S3 Logger for saving agent payloads and results
    Creates folder structure: deep_rsrch/dataz/outbound/{model}/{timestamp}/{folder_type}/
    Reads S3 bucket from config.yaml based on environment
    """
    
    def __init__(self, env: str):
        """
        Initialize S3 Logger
        Reads S3 configuration from config.yaml based on environment
        Environment mapping: dv/ts -> dev, pl -> uat, pr -> prod
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        """
        # Load S3 configuration from config.yaml
        config_loader = get_config_loader()
        s3_config = config_loader.get_s3_config(env)
        
        self.bucket_name = s3_config['bucket']
        self.base_path = s3_config.get('base_path', 'deep_rsrch/dataz/outbound')
        self.region = s3_config.get('region', 'us-east-1')
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        # Create timestamp folder at initialization (run level)
        self.timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Local save configuration
        self.local_base_path = "ETL"  # Base folder for local saves
        
        print(f"✅ S3Logger initialized for {env.upper()} environment")
        print(f"   Bucket: {self.bucket_name}")
        print(f"   Base Path: {self.base_path}")
        print(f"   Local Save Path: {self.local_base_path}")
        
    def save_to_local(self,
                      data: Dict[Any, Any],
                      filename: str,
                      model: str,
                      folder_type: str) -> bool:
        """
        Save data to local filesystem in the same structure as S3
        Structure: ETL/{model}/{timestamp}/{folder_type}/{filename}
        
        Args:
            data (dict): Data to save (will be converted to JSON)
            filename (str): Name of the file (e.g., 'correlation_request.json')
            model (str): Statistical model code (e.g., 'IP_AUTH', 'OP_AUTH')
            folder_type (str): Type of folder - 'payload', 'agents_results', or 'final_result'
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate folder type
            valid_folders = ['payload', 'agents_results', 'final_result']
            if folder_type not in valid_folders:
                print(f"Warning: Invalid folder_type '{folder_type}'. Must be one of: {valid_folders}")
                return False
            
            # Sanitize model name (replace spaces and special characters)
            model_sanitized = model.replace(" ", "_").replace("/", "_").replace("\\", "_")
            
            # Construct local path - ETL/model/timestamp/folder_type/
            local_dir = Path(self.local_base_path) / model_sanitized / self.timestamp_folder / folder_type
            
            # Create directories if they don't exist
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Full file path
            local_file_path = local_dir / filename
            
            # Convert data to JSON string
            json_string = json.dumps(data, indent=2, default=str)
            
            # Write to local file
            with open(local_file_path, 'w', encoding='utf-8') as f:
                f.write(json_string)
            
            print(f"💾 Saved locally: {local_file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving locally: {str(e)}")
            print(f"   Path: {local_file_path if 'local_file_path' in locals() else 'N/A'}")
            return False
    
    def save_to_s3(self, 
                   data: Dict[Any, Any], 
                   filename: str, 
                   model: str, 
                   folder_type: str) -> bool:
        """
        Save data to S3 in the appropriate folder structure
        Also saves to local filesystem as backup
        
        Args:
            data (dict): Data to save (will be converted to JSON)
            filename (str): Name of the file (e.g., 'correlation_request.json')
            model (str): Statistical model code (e.g., 'IP_AUTH', 'OP_AUTH')
            folder_type (str): Type of folder - 'payload', 'agents_results', or 'final_result'
            
        Returns:
            bool: True if successful, False otherwise
        """
        s3_success = False
        local_success = False
        
        try:
            # Validate folder type
            valid_folders = ['payload', 'agents_results', 'final_result']
            if folder_type not in valid_folders:
                print(f"Warning: Invalid folder_type '{folder_type}'. Must be one of: {valid_folders}")
                return False
            
            # Sanitize model name (replace spaces and special characters)
            model_sanitized = model.replace(" ", "_").replace("/", "_").replace("\\", "_")
            
            # Construct S3 key path - model/timestamp/folder_type/filename
            s3_key = f"{self.base_path}/{model_sanitized}/{self.timestamp_folder}/{folder_type}/{filename}"
            
            # Convert data to JSON string
            json_string = json.dumps(data, indent=2, default=str)
            
            # Upload to S3
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=json_string,
                    ContentType='application/json'
                )
                print(f"✅ Saved to S3: s3://{self.bucket_name}/{s3_key}")
                s3_success = True
            except Exception as e:
                print(f"❌ Error saving to S3: {str(e)}")
                print(f"   Bucket: {self.bucket_name}")
                print(f"   Key: {s3_key}")
            
            # Save to local filesystem as backup
            local_success = self.save_to_local(data, filename, model, folder_type)
            
            # Return True if at least one save succeeded
            return s3_success or local_success
            
        except Exception as e:
            print(f"❌ Error in save_to_s3: {str(e)}")
            return False
