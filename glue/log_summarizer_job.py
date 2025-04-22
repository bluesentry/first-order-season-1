import sys
import json
import boto3
import re
from urllib.parse import unquote_plus

# Check if running in AWS Glue environment
try:
    # For AWS Glue compatibility
    from awsglue.transforms import *
    from awsglue.utils import getResolvedOptions
    from pyspark.context import SparkContext
    from awsglue.context import GlueContext
    from awsglue.job import Job
    IN_GLUE_ENV = True
except ImportError:
    print("Not running in AWS Glue environment. Some functionality may be limited.")
    IN_GLUE_ENV = False

def clean_json_string(bad_json):
    """
    Clean up invalid control characters from JSON string
    
    Args:
        bad_json: JSON string that may contain invalid control characters
        
    Returns:
        Cleaned JSON string
    """
    return re.sub(r'[\x00-\x1F\x7F]', '', bad_json)

def list_log_files(s3_client, bucket_name, input_prefix):
    """
    Get all logs.json files under input_prefix
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        input_prefix: S3 prefix to search for logs.json files
        
    Returns:
        List of S3 keys for logs.json files
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=input_prefix)
    keys = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("logs.json"):
                keys.append(key)
    return keys

def process_logs(s3_client, bucket_name, input_prefix, output_prefix):
    """
    Process each logs.json file
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        input_prefix: S3 prefix to search for logs.json files
        output_prefix: S3 prefix to write cleaned logs.json files
        
    Returns:
        Dictionary with processing statistics
    """
    files = list_log_files(s3_client, bucket_name, input_prefix)
    print(f"📄 Found {len(files)} logs.json files to process.")
    
    stats = {
        "total_files": len(files),
        "processed_files": 0,
        "failed_files": 0,
        "cleaned_files": 0
    }

    for key in files:
        try:
            print(f"🔍 Processing: s3://{bucket_name}/{key}")
            raw = s3_client.get_object(Bucket=bucket_name, Key=key)["Body"].read().decode("utf-8", errors="replace")
            cleaned = clean_json_string(raw)
            
            # Check if cleaning was needed
            if cleaned != raw:
                stats["cleaned_files"] += 1
                print(f"🧹 Cleaned invalid control characters from {key}")
            
            # Parse JSON to validate it
            data = json.loads(cleaned)

            # Write cleaned JSON back to new location
            new_key = key.replace(input_prefix, output_prefix)
            s3_client.put_object(
                Bucket=bucket_name,
                Key=new_key,
                Body=json.dumps(data).encode("utf-8"),
                ContentType="application/json"
            )
            print(f"✅ Cleaned and saved to: s3://{bucket_name}/{new_key}")
            stats["processed_files"] += 1

        except Exception as e:
            print(f"❌ Failed to process {key}: {e}")
            stats["failed_files"] += 1
    
    return stats

def main():
    """
    Main function for AWS Glue job
    """
    # Initialize variables with default values
    bucket_name = "first-order-application-logs"
    input_prefix = "log-analysis/workflow-logs-for-llm/"
    output_prefix = "log-analysis-sanitized/"
    
    if IN_GLUE_ENV:
        # Parse job arguments when running in Glue
        args = getResolvedOptions(sys.argv, [
            'JOB_NAME',
            'bucket_name',
            'input_prefix',
            'output_prefix'
        ])
        
        # Initialize Glue context
        sc = SparkContext()
        glue_context = GlueContext(sc)
        job = Job(glue_context)
        job.init(args['JOB_NAME'], args)
        
        # Get parameters from args
        bucket_name = args.get('bucket_name', bucket_name)
        input_prefix = args.get('input_prefix', input_prefix)
        output_prefix = args.get('output_prefix', output_prefix)
    else:
        # When running locally, parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Process logs.json files in S3')
        parser.add_argument('--bucket_name', default=bucket_name, help='S3 bucket name')
        parser.add_argument('--input_prefix', default=input_prefix, help='S3 input prefix')
        parser.add_argument('--output_prefix', default=output_prefix, help='S3 output prefix')
        
        # Parse arguments
        args = parser.parse_args()
        bucket_name = args.bucket_name
        input_prefix = args.input_prefix
        output_prefix = args.output_prefix
    
    # Initialize S3 client
    s3_client = boto3.client("s3")
    
    # Process logs
    print(f"Starting log processing job")
    print(f"Bucket: {bucket_name}")
    print(f"Input prefix: {input_prefix}")
    print(f"Output prefix: {output_prefix}")
    
    stats = process_logs(s3_client, bucket_name, input_prefix, output_prefix)
    
    # Log completion
    print(f"Job completed successfully")
    print(f"Total files: {stats['total_files']}")
    print(f"Processed files: {stats['processed_files']}")
    print(f"Cleaned files: {stats['cleaned_files']}")
    print(f"Failed files: {stats['failed_files']}")
    
    # Commit the job if running in Glue
    if IN_GLUE_ENV:
        job.commit()

# Entry point for Glue job
if __name__ == "__main__":
    main()
