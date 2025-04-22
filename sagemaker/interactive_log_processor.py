#!/usr/bin/env python
# coding: utf-8

"""
Interactive Log Processing and Analysis

This script allows you to interactively test the log processing solution without deploying the Glue job.
You can run it in a SageMaker notebook instance or locally if you have the necessary permissions.

Setup Instructions:
1. Launch this script in a SageMaker notebook instance or locally
2. Make sure your IAM role has access to the S3 bucket containing your logs
3. Run the script to process and analyze your logs
4. Modify the code as needed to customize the analysis
"""

import sys
import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os
import io
import gzip
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

# For embedding generation and vector storage
USE_ML_FEATURES = True
try:
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss
    import pickle
    print("ML dependencies successfully imported.")
except ImportError:
    print("Warning: ML dependencies not found. Running in basic mode without embedding features.")
    print("To enable full features, install the required packages manually:")
    print("1. In SageMaker terminal: pip install torch==1.10.0 sentence-transformers==2.2.2 faiss-cpu==1.7.4")
    print("2. Or use conda: conda install -c pytorch pytorch==1.10.0 faiss-cpu==1.7.4")
    print("3. Then: pip install sentence-transformers==2.2.2")
    USE_ML_FEATURES = False
#!/usr/bin/env python
# coding: utf-8

"""
Interactive Log Processing and Analysis

This script allows you to interactively test the log processing solution without deploying the Glue job.
You can run it in a SageMaker notebook instance or locally if you have the necessary permissions.

Setup Instructions:
1. Launch this script in a SageMaker notebook instance or locally
2. Make sure your IAM role has access to the S3 bucket containing your logs
3. Run the script to process and analyze your logs
4. Modify the code as needed to customize the analysis
"""

import sys
import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os
import io
import gzip
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

#!/usr/bin/env python
# coding: utf-8

"""
Interactive Log Processing and Analysis

This script allows you to interactively test the log processing solution without deploying the Glue job.
You can run it in a SageMaker notebook instance or locally if you have the necessary permissions.

Setup Instructions:
1. Launch this script in a SageMaker notebook instance or locally
2. Make sure your IAM role has access to the S3 bucket containing your logs
3. Run the script to process and analyze your logs
4. Modify the code as needed to customize the analysis
"""

import sys
import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os
import io
import gzip
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

# For embedding generation
try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "faiss-cpu", "torch"])
    import torch
    from sentence_transformers import SentenceTransformer

# For vector storage
import faiss
import pickle

# Configure plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Initialize AWS clients
s3 = boto3.client('s3')
sns_client = boto3.client('sns')

# S3 Configuration
S3_BUCKET = "first-order-application-logs"  # Your S3 bucket name
S3_INPUT_PATH = f"s3://{S3_BUCKET}/fluent-bit-logs/"  # Path to your log files
S3_OUTPUT_PATH = f"s3://{S3_BUCKET}/log-analysis/"  # Path to save analysis results

# Model Configuration
MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer model
VECTOR_DIM = 384  # Embedding dimension

# Analysis Configuration
MIN_GROUP_SIZE = 3  # Minimum number of logs to consider a pattern
MAX_LOGS_TO_PROCESS = 1000  # Maximum number of logs to process (set to None for all logs)


class InteractiveLogProcessor:
    """
    A class to process, analyze, and summarize logs interactively.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2", 
                 vector_dim: int = 384,
                 s3_model_path: Optional[str] = None):
        """
        Initialize the log processor
        
        Args:
            model_name: Name of the sentence transformer model
            vector_dim: Dimension of the embedding vectors
            s3_model_path: Path to load model from S3
        """
        # Initialize S3 client
        self.s3 = boto3.client('s3')
        
        # Initialize embedding model
        self.model_name = model_name
        self.vector_dim = vector_dim
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize FAISS index
        self.index = None
        self.log_data = []
        
        # Load existing model if path provided
        if s3_model_path:
            self.load_from_s3(s3_model_path)
        else:
            # Create new index
            self.index = faiss.IndexFlatL2(vector_dim)
    
    def parse_logs_from_s3(self, s3_path: str, max_logs: Optional[int] = None) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Parse logs from S3 location (supports JSON, GZIP)
        
        Args:
            s3_path: S3 URI to the logs
            max_logs: Maximum number of logs to process
            
        Returns:
            Tuple of (parsed logs, features DataFrame)
        """
        parsed_logs = []
        
        # Parse S3 URI
        parsed_uri = urlparse(s3_path)
        bucket = parsed_uri.netloc
        key = parsed_uri.path.lstrip('/')
        
        print(f"Searching for logs in s3://{bucket}/{key}")
        
        # Handle directory with nested structure (year/month/day/hour/minute)
        if not key.endswith('.json') and not key.endswith('.gz'):
            # List all objects recursively in the directory
            paginator = self.s3.get_paginator('list_objects_v2')
            files = []
            
            # Iterate through all pages of results
            for page in paginator.paginate(Bucket=bucket, Prefix=key):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        obj_key = obj['Key']
                        # Only process .json or .gz files
                        if obj_key.endswith('.json') or obj_key.endswith('.gz') or '.json.gz' in obj_key:
                            files.append(obj_key)
            
            print(f"Found {len(files)} log files")
        else:
            # Single file
            files = [key]
            
        # Process each file
        for file_key in files:
            if not (file_key.endswith('.json') or file_key.endswith('.gz')):
                continue
                
            try:
                # Get the object from S3
                print(f"Processing file: s3://{bucket}/{file_key}")
                obj = self.s3.get_object(Bucket=bucket, Key=file_key)
                
                if file_key.endswith('.gz') or '.json.gz' in file_key:
                    # Decompress and decode
                    with gzip.GzipFile(fileobj=io.BytesIO(obj['Body'].read())) as f:
                        content = f.read().decode('utf-8')
                else:
                    # Just decode
                    content = obj['Body'].read().decode('utf-8')
            except Exception as e:
                print(f"Error processing file {file_key}: {e}")
                continue
                
            # Process each line as a JSON record
            for line in content.splitlines():
                try:
                    log_entry = json.loads(line.strip())
                    parsed_logs.append(log_entry)
                    
                    # Check if we've reached the maximum number of logs
                    if max_logs is not None and len(parsed_logs) >= max_logs:
                        print(f"Reached maximum number of logs to process: {max_logs}")
                        break
                except json.JSONDecodeError:
                    print(f"Failed to parse line: {line[:100]}...")
                    continue
            
            # Check if we've reached the maximum number of logs
            if max_logs is not None and len(parsed_logs) >= max_logs:
                break
                    
        print(f"Parsed {len(parsed_logs)} logs")
        
        # Extract features
        features_df = self.extract_features(parsed_logs)
        
        return parsed_logs, features_df
    
    def extract_features(self, logs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract relevant features from parsed logs
        
        Args:
            logs: List of parsed log entries
            
        Returns:
            DataFrame with extracted features
        """
        features = []
        
        for log in logs:
            # Extract basic log information
            log_entry = {
                'timestamp': log.get('date'),
                'message': log.get('log'),
                'log_level': None,
                'component': None,
                'action': None,
                'status': None,
                'error_type': None,
                'error_details': None,
                'pod_name': None,
                'namespace': None,
                'host': None,
                'container_name': None,
                'container_image': None
            }
            
            # Extract log level, component and action from message
            if log.get('log'):
                log_msg = log.get('log')
                
                # Extract log level
                level_match = re.search(r'\[(info|warn|error|debug)\]', log_msg, re.IGNORECASE)
                if level_match:
                    log_entry['log_level'] = level_match.group(1).lower()
                
                # Extract component
                component_match = re.search(r'\[([^:]+):([^:]+):([^\]]+)\]', log_msg)
                if component_match:
                    log_entry['component'] = f"{component_match.group(1)}:{component_match.group(2)}:{component_match.group(3)}"
                
                # Extract action and status
                if "Successfully" in log_msg and "uploaded" in log_msg:
                    log_entry['action'] = "upload"
                    log_entry['status'] = "success"
                elif "Failed to upload" in log_msg:
                    log_entry['action'] = "upload"
                    log_entry['status'] = "failure"
                    
                    # Extract error details
                    error_match = re.search(r'Failed to upload.*: (\w+)', log_msg)
                    if error_match:
                        log_entry['error_type'] = error_match.group(1)
                        log_entry['error_details'] = log_msg.split("Failed to upload")[1].strip()
                elif "Connection timeout" in log_msg:
                    log_entry['action'] = "connect"
                    log_entry['status'] = "failure"
                    log_entry['error_type'] = "Timeout"
                    
                    # Extract timeout details
                    timeout_match = re.search(r'timeout after (\d+)s', log_msg)
                    if timeout_match:
                        log_entry['error_details'] = f"Timeout after {timeout_match.group(1)} seconds"
                elif "Invalid JSON" in log_msg:
                    log_entry['action'] = "parse"
                    log_entry['status'] = "failure"
                    log_entry['error_type'] = "InvalidFormat"
                    log_entry['error_details'] = "Invalid JSON format"
                elif "Scanning log file" in log_msg:
                    log_entry['action'] = "scan"
                    log_entry['status'] = "info"
                elif "File rotated" in log_msg:
                    log_entry['action'] = "rotate"
                    log_entry['status'] = "info"
                elif "Retrying" in log_msg and "upload" in log_msg:
                    log_entry['action'] = "retry"
                    log_entry['status'] = "warning"
                    
                    # Extract retry details
                    retry_match = re.search(r'attempt (\d+)/(\d+)', log_msg)
                    if retry_match:
                        log_entry['error_details'] = f"Attempt {retry_match.group(1)} of {retry_match.group(2)}"
                elif "Started" in log_msg:
                    log_entry['action'] = "start"
                    log_entry['status'] = "info"
                elif "Shutting down" in log_msg:
                    log_entry['action'] = "shutdown"
                    log_entry['status'] = "info"
            
            # Extract Kubernetes metadata if available
            if 'kubernetes' in log:
                k8s = log.get('kubernetes', {})
                log_entry['pod_name'] = k8s.get('pod_name')
                log_entry['namespace'] = k8s.get('namespace_name')
                log_entry['host'] = k8s.get('host')
                log_entry['container_name'] = k8s.get('container_name')
                log_entry['container_image'] = k8s.get('container_image')
            
            features.append(log_entry)
            
        return pd.DataFrame(features)
    
    def generate_embeddings(self, df: pd.DataFrame, text_column: str = 'message') -> np.ndarray:
        """
        Generate embeddings for log messages
        
        Args:
            df: DataFrame containing logs
            text_column: Column containing text to embed
            
        Returns:
            NumPy array of embeddings
        """
        texts = df[text_column].fillna('').tolist()
        
        # Process in batches to avoid memory issues
        batch_size = 128
        all_embeddings = []
        
        print(f"Generating embeddings for {len(texts)} logs in batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            # Print progress
            if (i + batch_size) % 500 == 0 or i + batch_size >= len(texts):
                print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} logs")
            
        # Combine batches
        embeddings = np.vstack(all_embeddings)
        return embeddings
    
    def add_to_index(self, df: pd.DataFrame, embeddings: np.ndarray):
        """
        Add logs and their embeddings to the index
        
        Args:
            df: DataFrame containing logs
            embeddings: NumPy array of embeddings
        """
        if len(df) != embeddings.shape[0]:
            raise ValueError("Number of logs and embeddings must match")
            
        # Initialize index if not already done
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.vector_dim)
            
        # Store original data
        start_idx = len(self.log_data)
        self.log_data.extend(df.to_dict('records'))
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        print(f"Added {len(df)} logs to index. Total logs: {len(self.log_data)}")
    
    def identify_critical_errors(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify critical errors in logs
        
        Args:
            df: DataFrame containing logs
            
        Returns:
            List of critical errors
        """
        # Define critical error patterns
        critical_patterns = [
            # Access denied errors
            {'pattern': 'AccessDenied', 'severity': 'high', 'category': 'permission', 
             'description': 'S3 access denied error', 'column': 'error_type'},
            # Connection timeouts
            {'pattern': 'Timeout', 'severity': 'medium', 'category': 'network', 
             'description': 'Connection timeout', 'column': 'error_type'},
            # Invalid formats
            {'pattern': 'InvalidFormat', 'severity': 'medium', 'category': 'data', 
             'description': 'Invalid data format', 'column': 'error_type'},
            # Retry attempts
            {'pattern': 'retry', 'severity': 'low', 'category': 'operation', 
             'description': 'Operation retry', 'column': 'action'}
        ]
        
        # Find matches
        critical_errors = []
        
        for pattern in critical_patterns:
            column = pattern['column']
            matches = df[df[column] == pattern['pattern']]
            
            if len(matches) > 0:
                for _, row in matches.iterrows():
                    error = {
                        'timestamp': row.get('timestamp'),
                        'message': row.get('message'),
                        'pod_name': row.get('pod_name'),
                        'namespace': row.get('namespace'),
                        'severity': pattern['severity'],
                        'category': pattern['category'],
                        'description': pattern['description'],
                        'details': row.get('error_details'),
                        'count': 1
                    }
                    critical_errors.append(error)
        
        # Group similar errors
        grouped_errors = {}
        for error in critical_errors:
            key = f"{error['severity']}_{error['category']}_{error['description']}"
            if key in grouped_errors:
                grouped_errors[key]['count'] += 1
                # Keep the most recent occurrence
                if error['timestamp'] > grouped_errors[key]['timestamp']:
                    grouped_errors[key]['timestamp'] = error['timestamp']
                    grouped_errors[key]['message'] = error['message']
                    grouped_errors[key]['details'] = error['details']
            else:
                grouped_errors[key] = error
        
        return list(grouped_errors.values())
    
    def identify_patterns(self, df: pd.DataFrame, min_group_size: int = 3) -> Dict[str, Any]:
        """
        Identify patterns in logs based on common attributes
        
        Args:
            df: DataFrame containing logs
            min_group_size: Minimum number of logs to consider a pattern
            
        Returns:
            Dictionary of identified patterns
        """
        patterns = {}
        
        # Group by component and action
        for (component, action), group_df in df.groupby(['component', 'action']):
            if pd.isna(component) or pd.isna(action) or len(group_df) < min_group_size:
                continue
                
            # Get the most common status
            status = group_df['status'].mode().iloc[0] if not group_df['status'].isna().all() else 'unknown'
            
            # Calculate time intervals
            if 'timestamp' in group_df.columns and len(group_df) > 1:
                try:
                    # Convert to datetime if it's not already
                    if not pd.api.types.is_datetime64_any_dtype(group_df['timestamp']):
                        group_df['timestamp'] = pd.to_datetime(group_df['timestamp'])
                        
                    # Sort and calculate intervals
                    sorted_df = group_df.sort_values('timestamp')
                    sorted_df['time_diff'] = sorted_df['timestamp'].diff().dt.total_seconds()
                    
                    median_interval = sorted_df['time_diff'].median()
                    mean_interval = sorted_df['time_diff'].mean()
                except Exception as e:
                    print(f"Error calculating time intervals: {e}")
                    median_interval = None
                    mean_interval = None
            else:
                median_interval = None
                mean_interval = None
            
            # Create pattern entry
            pattern_key = f"{component}_{action}"
            patterns[pattern_key] = {
                'component': component,
                'action': action,
                'status': status,
                'count': len(group_df),
                'median_interval_seconds': float(median_interval) if pd.notna(median_interval) else None,
                'mean_interval_seconds': float(mean_interval) if pd.notna(mean_interval) else None,
                'sample_logs': group_df.head(3)['message'].tolist()
            }
        
        return patterns
    
    def generate_suggested_fixes(self, critical_errors: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Generate suggested fixes for critical errors
        
        Args:
            critical_errors: List of critical errors
            
        Returns:
            Dictionary of suggested fixes
        """
        # Common fixes for known error types
        common_fixes = {
            'AccessDenied': [
                "Check IAM permissions for the FluentBit service account",
                "Verify S3 bucket policy allows write access from the cluster's IP range",
                "Ensure KMS key permissions are properly configured if using SSE-KMS"
            ],
            'Timeout': [
                "Check network connectivity between the cluster and S3 endpoint",
                "Verify VPC endpoints are properly configured",
                "Consider increasing the timeout setting in FluentBit configuration"
            ],
            'InvalidFormat': [
                "Review the log format configuration in FluentBit",
                "Check for malformed JSON in application logs",
                "Add a parser filter to handle the specific log format"
            ]
        }
        
        suggested_fixes = {}
        
        for error in critical_errors:
            if error['severity'] == 'low':
                continue
                
            key = f"{error['severity']}_{error['category']}_{error['description']}"
            
            # Get fixes based on error type
            if error['error_type'] in common_fixes:
                suggested_fixes[key] = common_fixes[error['error_type']]
            else:
                suggested_fixes[key] = ["No specific fixes available for this error type."]
        
        return suggested_fixes
    
    def generate_summary(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a human-readable summary of logs
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            Dictionary containing the summary
        """
        # Identify critical errors and patterns
        critical_errors = self.identify_critical_errors(features_df)
        patterns = self.identify_patterns(features_df)
        
        # Generate suggested fixes
        suggested_fixes = self.generate_suggested_fixes(critical_errors)
        
        # Create a summary dictionary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_logs_analyzed': len(features_df),
            'critical_errors': critical_errors,
            'patterns': patterns,
            'suggested_fixes': suggested_fixes,
            'summary_text': ""
        }
        
        # Generate a text summary
        summary_text = f"Log Analysis Summary ({len(features_df)} logs)\n\n"
        
        # Add critical errors section
        if critical_errors:
            summary_text += f"Critical Errors ({len(critical_errors)}):\n"
            for error in critical_errors:
                summary_text += f"- {error['severity'].upper()} {error['category']}: {error['description']} (count: {error['count']})\n"
                summary_text += f"  Example: {error['message']}\n"
                if error['details']:
                    summary_text += f"  Details: {error['details']}\n"
                
                # Add suggested fixes
                key = f"{error['severity']}_{error['category']}_{error['description']}"
                if key in suggested_fixes:
                    summary_text += "  Suggested fixes:\n"
                    for fix in suggested_fixes[key]:
                        summary_text += f"    * {fix}\n"
                
                summary_text += "\n"
        else:
            summary_text += "No critical errors detected.\n\n"
        
        # Add patterns section
        if patterns:
            summary_text += f"Log Patterns ({len(patterns)}):\n"
            for key, pattern in patterns.items():
                summary_text += f"- {pattern['component']} {pattern['action']} ({pattern['count']} logs)\n"
                if pattern['sample_logs']:
                    summary_text += f"  Example: {pattern['sample_logs'][0]}\n"
                if pattern['median_interval_seconds']:
                    summary_text += f"  Median interval: {pattern['median_interval_seconds']:.2f} seconds\n"
                summary_text += "\n"
        else:
            summary_text += "No significant patterns detected.\n\n"
        
        summary['summary_text'] = summary_text
        
        return summary
    
    def save_summary_to_s3(self, summary: Dict[str, Any], s3_output_path: str):
        """
        Save the summary to S3
        
        Args:
            summary: Summary dictionary
            s3_output_path: S3 URI to save the summary
        """
        # Parse S3 URI
        parsed_uri = urlparse(s3_output_path)
        bucket = parsed_uri.netloc
        key = parsed_uri.path.lstrip('/')
        
        # Convert to JSON
        summary_json = json.dumps(summary, default=str)
        
        # Upload to S3
        self.s3.put_object(
            Body=summary_json,
            Bucket=bucket,
            Key=key
        )
        
        print(f"Saved summary to s3://{bucket}/{key}")
    
    def search_similar_logs(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for logs similar to the query
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar logs
        """
        if self.index is None or not self.log_data:
            return []
            
        # Generate embedding for query
        query_embedding = self.model.encode([query])[0].reshape(1, -1).astype(np.float32)
        
        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.log_data):  # Ensure index is valid
                result = self.log_data[idx].copy()
                result['similarity_score'] = float(distances[0][i])
                results.append(result)
                
        return results
    
    def visualize_log_patterns(self, df: pd.DataFrame):
        """
        Visualize log patterns
        
        Args:
            df: DataFrame containing logs
        """
        # Count logs by component
        if 'component' in df.columns:
            component_counts = df['component'].value_counts().head(10)
            plt.figure(figsize=(12, 6))
            sns.barplot(x=component_counts.index, y=component_counts.values)
            plt.title('Top 10 Components by Log Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        
        # Count logs by action
        if 'action' in df.columns:
            action_counts = df['action'].value_counts().head(10)
            plt.figure(figsize=(12, 6))
            sns.barplot(x=action_counts.index, y=action_counts.values)
            plt.title('Top 10 Actions by Log Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        
        # Count logs by status
        if 'status' in df.columns:
            status_counts = df['status'].value_counts()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=status_counts.index, y=status_counts.values)
            plt.title('Log Count by Status')
            plt.tight_layout()
            plt.show()
        
        # Count logs by log level
        if 'log_level' in df.columns:
            level_counts = df['log_level'].value_counts()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=level_counts.index, y=level_counts.values)
            plt.title('Log Count by Log Level')
            plt.tight_layout()
            plt.show()
    
    def visualize_critical_errors(self, critical_errors: List[Dict[str, Any]]):
        """
        Visualize critical errors
        
        Args:
            critical_errors: List of critical errors
        """
        if not critical_errors:
            print("No critical errors to visualize.")
            return
        
        # Count errors by severity
        severity_counts = {}
        for error in critical_errors:
            severity = error['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + error['count']
        
        plt.figure(figsize=(10, 6))
        colors = {'high': 'red', 'medium': 'orange', 'low': 'yellow'}
        sns.barplot(x=list(severity_counts.keys()), y=list(severity_counts.values()), 
                   palette=[colors.get(s, 'blue') for s in severity_counts.keys()])
        plt.title('Critical Errors by Severity')
        plt.tight_layout()
        plt.show()
        
        # Count errors by category
        category_counts = {}
        for error in critical_errors:
            category = error['category']
            category_counts[category] = category_counts.get(category, 0) + error['count']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()))
        plt.title('Critical Errors by Category')
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to run the interactive log processor
    """
    print("Starting Interactive Log Processor")
    
    # Initialize processor
    processor = InteractiveLogProcessor(
        model_name=MODEL_NAME,
        vector_dim=VECTOR_DIM
    )
    
    # Parse logs from S3
    logs, features_df = processor.parse_logs_from_s3(S3_INPUT_PATH, max_logs=MAX_LOGS_TO_PROCESS)
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(f"Total logs: {len(logs)}")
    print(f"Unique components: {features_df['component'].nunique()}")
    print(f"Unique actions: {features_df['action'].nunique()}")
    print(f"Unique log levels: {features_df['log_level'].nunique()}")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = processor.generate_embeddings(features_df)
    
    # Add to index
    processor.add_to_index(features_df, embeddings)
    
    # Identify critical errors
    print("\nIdentifying critical errors...")
    critical_errors = processor.identify_critical_errors(features_df)
    print(f"Found {len(critical_errors)} critical errors")
    
    # Identify patterns
    print("\nIdentifying patterns...")
    patterns = processor.identify_patterns(features_df, min_group_size=MIN_GROUP_SIZE)
    print(f"Found {len(patterns)} patterns")
    
    # Generate summary
    print("\nGenerating summary...")
    summary = processor.generate_summary(features_df)
    
    # Print summary
    print("\n" + "="*80)
    print(summary['summary_text'])
    print("="*80)
    
    # Save summary to S3
    summary_path = f"{S3_OUTPUT_PATH}/summary/interactive_summary.json"
    print(f"\nSaving summary to {summary_path}...")
    processor.save_summary_to_s3(summary, summary_path)
    
    # Visualize log patterns
    print("\nVisualizing log patterns...")
    processor.visualize_log_patterns(features_df)
    
    # Visualize critical errors
    print("\nVisualizing critical errors...")
    processor.visualize_critical_errors(critical_errors)
    
    # Example of searching for similar logs
    print("\nExample of searching for similar logs:")
    query = "Failed to upload"
    similar_logs = processor.search_similar_logs(query, k=3)
    print(f"Logs similar to '{query}':")
    for i, log in enumerate(similar_logs, 1):
        print(f"{i}. {log.get('message')} (Score: {log.get('similarity_score'):.4f})")
    
    print("\nInteractive Log Processor completed successfully!")


if __name__ == "__main__":
    main()
