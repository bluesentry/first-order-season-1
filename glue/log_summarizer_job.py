import sys
import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

# For AWS Glue compatibility
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

# For embedding generation - packaged as a wheel in S3
import torch
from sentence_transformers import SentenceTransformer

# For vector storage - FAISS index will be stored in S3
import faiss
import pickle

class LogSummarizer:
    """
    A class to summarize logs, identify critical errors, and suggest fixes.
    This extends the GlueLogProcessor with summarization capabilities.
    """
    
    def __init__(self, 
                 glue_context,
                 model_name: str = "all-MiniLM-L6-v2", 
                 vector_dim: int = 384,
                 s3_model_path: Optional[str] = None):
        """
        Initialize the log summarizer
        
        Args:
            glue_context: Glue context
            model_name: Name of the sentence transformer model
            vector_dim: Dimension of the embedding vectors
            s3_model_path: Path to load model from S3
        """
        self.glue_context = glue_context
        self.spark = glue_context.spark_session
        
        # Initialize S3 client
        self.s3 = boto3.client('s3')
        
        # Initialize embedding model
        self.model_name = model_name
        self.vector_dim = vector_dim
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
    
    def parse_logs_from_s3(self, s3_path: str) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Parse logs from S3 location (supports JSON, GZIP)
        
        Args:
            s3_path: S3 URI to the logs
            
        Returns:
            Tuple of (parsed logs, features DataFrame)
        """
        parsed_logs = []
        
        # Parse S3 URI
        parsed_uri = urlparse(s3_path)
        bucket = parsed_uri.netloc
        key = parsed_uri.path.lstrip('/')
        
        # Handle directory vs file
        if not key.endswith('.json') and not key.endswith('.gz'):
            # List objects in directory
            response = self.s3.list_objects_v2(Bucket=bucket, Prefix=key)
            files = [obj['Key'] for obj in response.get('Contents', [])]
        else:
            # Single file
            files = [key]
            
        # Process each file
        for file_key in files:
            if not (file_key.endswith('.json') or file_key.endswith('.gz')):
                continue
                
            # Get the object from S3
            obj = self.s3.get_object(Bucket=bucket, Key=file_key)
            
            if file_key.endswith('.gz'):
                # Decompress and decode
                with gzip.GzipFile(fileobj=io.BytesIO(obj['Body'].read())) as f:
                    content = f.read().decode('utf-8')
            else:
                # Just decode
                content = obj['Body'].read().decode('utf-8')
                
            # Process each line as a JSON record
            for line in content.splitlines():
                try:
                    log_entry = json.loads(line.strip())
                    parsed_logs.append(log_entry)
                except json.JSONDecodeError:
                    print(f"Failed to parse line: {line}")
                    continue
                    
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
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            all_embeddings.append(batch_embeddings)
            
        # Combine batches
        embeddings = np.vstack(all_embeddings)
        return embeddings
    
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
    
    def create_feature_dynamic_frame(self, df: pd.DataFrame) -> DynamicFrame:
        """
        Convert pandas DataFrame to Glue DynamicFrame for output
        
        Args:
            df: DataFrame to convert
            
        Returns:
            Glue DynamicFrame
        """
        # Convert pandas DataFrame to Spark DataFrame
        spark_df = self.spark.createDataFrame(df)
        
        # Convert to DynamicFrame
        dynamic_frame = DynamicFrame.fromDF(spark_df, self.glue_context, "features")
        
        return dynamic_frame
    
    def save_to_s3(self, s3_output_path: str):
        """
        Save the processor state to S3
        
        Args:
            s3_output_path: S3 URI to save the processor state
        """
        # Parse S3 URI
        parsed_uri = urlparse(s3_output_path)
        bucket = parsed_uri.netloc
        prefix = parsed_uri.path.lstrip('/')
        
        if not prefix.endswith('/'):
            prefix += '/'
        
        # Serialize the index
        faiss_buffer = io.BytesIO()
        faiss.write_index(self.index, faiss_buffer)
        faiss_buffer.seek(0)
        
        # Save FAISS index to S3
        self.s3.upload_fileobj(
            faiss_buffer,
            bucket,
            f"{prefix}log_index.faiss"
        )
        
        # Serialize and save log data
        log_data_buffer = io.BytesIO()
        pickle.dump(self.log_data, log_data_buffer)
        log_data_buffer.seek(0)
        
        self.s3.upload_fileobj(
            log_data_buffer,
            bucket,
            f"{prefix}log_data.pkl"
        )
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'vector_dim': self.vector_dim,
            'num_logs': len(self.log_data),
            'created_at': datetime.now().isoformat()
        }
        
        metadata_buffer = io.BytesIO(json.dumps(metadata).encode('utf-8'))
        
        self.s3.upload_fileobj(
            metadata_buffer,
            bucket,
            f"{prefix}metadata.json"
        )
        
        print(f"Saved processor state to s3://{bucket}/{prefix}")
    
    def load_from_s3(self, s3_path: str):
        """
        Load the processor state from S3
        
        Args:
            s3_path: S3 URI to load the processor state from
        """
        # Parse S3 URI
        parsed_uri = urlparse(s3_path)
        bucket = parsed_uri.netloc
        prefix = parsed_uri.path.lstrip('/')
        
        if not prefix.endswith('/'):
            prefix += '/'
        
        # Load FAISS index
        faiss_buffer = io.BytesIO()
        self.s3.download_fileobj(bucket, f"{prefix}log_index.faiss", faiss_buffer)
        faiss_buffer.seek(0)
        self.index = faiss.read_index(faiss_buffer)
        
        # Load log data
        log_data_buffer = io.BytesIO()
        self.s3.download_fileobj(bucket, f"{prefix}log_data.pkl", log_data_buffer)
        log_data_buffer.seek(0)
        self.log_data = pickle.load(log_data_buffer)
        
        print(f"Loaded processor state from s3://{bucket}/{prefix} with {len(self.log_data)} logs")


# Main function for AWS Glue job
def process_logs_glue_job():
    # Parse job arguments
    args = getResolvedOptions(sys.argv, [
        'JOB_NAME',
        's3_input_path',
        's3_output_path',
        's3_model_path',
        'save_model',
        'notification_topic'
    ])
    
    # Initialize Glue context
    sc = SparkContext()
    glue_context = GlueContext(sc)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(args['JOB_NAME'], args)
    
    # Get S3 paths
    s3_input_path = args['s3_input_path']
    s3_output_path = args['s3_output_path']
    s3_model_path = args.get('s3_model_path', None)
    save_model = args.get('save_model', 'true').lower() == 'true'
    notification_topic = args.get('notification_topic', None)
    
    # Initialize summarizer
    summarizer = LogSummarizer(
        glue_context=glue_context,
        s3_model_path=s3_model_path
    )
    
    # Process logs
    logs, features_df = summarizer.parse_logs_from_s3(s3_input_path)
    
    # Generate embeddings
    print(f"Generating embeddings for {len(features_df)} logs")
    embeddings = summarizer.generate_embeddings(features_df)
    
    # Add to index
    summarizer.add_to_index(features_df, embeddings)
    
    # Generate summary
    summary = summarizer.generate_summary(features_df)
    
    # Save output
    # Convert to DynamicFrame and write features
    features_dyf = summarizer.create_feature_dynamic_frame(features_df)
    glue_context.write_dynamic_frame.from_options(
        frame=features_dyf,
        connection_type="s3",
        connection_options={"path": f"{s3_output_path}/features"},
        format="parquet"
    )
    
    # Save summary
    summarizer.save_summary_to_s3(summary, f"{s3_output_path}/summary/summary.json")
    
    # Save model if requested
    if save_model:
        model_output_path = f"{s3_output_path}/model"
        summarizer.save_to_s3(model_output_path)
    
    # Send notification if topic provided
    if notification_topic:
        sns = boto3.client('sns')
        
        # Create a simplified summary for the notification
        notification_summary = {
            'timestamp': summary['timestamp'],
            'total_logs_analyzed': summary['total_logs_analyzed'],
            'critical_errors_count': len(summary['critical_errors']),
            'patterns_count': len(summary['patterns']),
            'summary_url': f"s3://{s3_output_path}/summary/summary.json"
        }
        
        # Add critical errors
        if summary['critical_errors']:
            notification_summary['critical_errors'] = []
            for error in summary['critical_errors']:
                if error['severity'] != 'low':  # Only include medium and high severity errors
                    notification_summary['critical_errors'].append({
                        'severity': error['severity'],
                        'category': error['category'],
                        'description': error['description'],
                        'count': error['count']
                    })
        
        # Send notification
        sns.publish(
            TopicArn=notification_topic,
            Subject=f"Log Analysis Summary - {datetime.now().strftime('%Y-%m-%d')}",
            Message=json.dumps(notification_summary, indent=2, default=str)
        )
    
    # Log completion
    print(f"Processed {len(logs)} logs, identified {len(summary['critical_errors'])} critical errors and {len(summary['patterns'])} patterns")
    print(f"Output written to {s3_output_path}")
    
    # Commit the job
    job.commit()


# Entry point for Glue job
if __name__ == "__main__":
    process_logs_glue_job()
