# Kubernetes Log Processing and Analysis

This project provides a comprehensive solution for processing, analyzing, and summarizing Kubernetes application logs. It uses AWS services to collect logs from Kubernetes applications, store them in S3, process them with Glue, and generate human-readable summaries with suggested fixes for critical errors.

## Architecture Overview

The solution consists of the following components:

1. **Log Collection**: FluentBit collects logs from Kubernetes applications and sends them to S3
2. **Log Storage**: S3 bucket stores logs in a hierarchical structure (year/month/day/hour/minute)
3. **Log Processing**: AWS Glue job processes logs, extracts features, and identifies patterns
4. **Log Analysis**: Machine learning techniques identify critical errors and suggest fixes
5. **Visualization**: CloudWatch dashboard displays log metrics and SageMaker notebooks provide interactive analysis

## Components

### 1. Glue Job for Log Processing

The Glue job (`glue/log_summarizer_job.py`) is the core of the solution. It:

- Processes logs from S3 in a nested folder structure
- Extracts features from logs (components, actions, errors, etc.)
- Generates embeddings for log messages using sentence transformers
- Identifies critical errors and patterns in logs
- Suggests fixes for common issues
- Generates human-readable summaries
- Sends notifications via SNS

### 2. SageMaker Notebooks for Interactive Analysis

Two SageMaker notebooks are provided for interactive analysis:

- `sagemaker/log_summarizer_simple.ipynb`: A simple notebook for basic log analysis
- `sagemaker/interactive_log_processor.py`: A Python script for interactive log processing and analysis

### 3. Terraform Configuration for Infrastructure

The Terraform configuration (`terraform/glue_job.tf`) sets up the necessary AWS resources:

- Glue job for log processing
- SNS topic for notifications
- CloudWatch dashboard for monitoring
- IAM roles and policies for access control

## Getting Started

### Prerequisites

- AWS account with appropriate permissions
- Terraform installed
- Python 3.8+ installed
- Kubernetes cluster with FluentBit configured to send logs to S3

### Deployment

1. **Deploy the infrastructure**:

   ```bash
   cd terraform
   terraform init
   terraform apply
   ```

2. **Upload dependencies to S3**:

   ```bash
   # Create a virtual environment and install dependencies
   python -m venv venv
   source venv/bin/activate
   pip install sentence-transformers faiss-cpu torch
   
   # Download the wheel files
   pip download sentence-transformers faiss-cpu --dest ./wheels
   
   # Upload to S3
   aws s3 cp ./wheels/sentence-transformers-2.2.2-py3-none-any.whl s3://first-order-application-logs/glue-dependencies/
   aws s3 cp ./wheels/faiss-1.7.4-py3-none-any.whl s3://first-order-application-logs/glue-dependencies/
   ```

3. **Subscribe to notifications**:

   - Go to the AWS SNS console
   - Find the topic `first-order-log-analysis-notifications`
   - Create a subscription (email, SMS, or other endpoint)

### Interactive Analysis

For interactive analysis, you can use the SageMaker notebooks:

1. **Launch a SageMaker notebook instance**:

   - Go to the AWS SageMaker console
   - Create a new notebook instance
   - Ensure the IAM role has access to the S3 bucket with logs

2. **Upload and run the interactive log processor**:

   - Upload `sagemaker/interactive_log_processor.py` to the notebook instance
   - Open a terminal and run:
     ```bash
     python interactive_log_processor.py
     ```
   - Or create a new notebook and import the script:
     ```python
     %run interactive_log_processor.py
     ```

3. **Customize the analysis**:

   - Modify the script to adjust the log processing parameters
   - Add new error patterns or suggested fixes
   - Create custom visualizations

## Customization

### Adding New Error Patterns

To add new error patterns, modify the `identify_critical_errors` method in `glue/log_summarizer_job.py`:

```python
critical_patterns = [
    # Existing patterns...
    
    # New pattern
    {'pattern': 'YourErrorPattern', 'severity': 'high', 'category': 'your_category', 
     'description': 'Your error description', 'column': 'error_type'},
]
```

### Adding New Suggested Fixes

To add new suggested fixes, modify the `generate_suggested_fixes` method:

```python
common_fixes = {
    # Existing fixes...
    
    # New fixes
    'YourErrorType': [
        "Suggested fix 1",
        "Suggested fix 2",
        "Suggested fix 3"
    ]
}
```

### Adjusting Log Processing Parameters

You can adjust the log processing parameters in the Terraform configuration:

```hcl
default_arguments = {
  # Existing arguments...
  
  # Custom arguments
  "--your-custom-argument" = "your-value"
}
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**:
   - Ensure the required Python dependencies are uploaded to S3
   - Check the Glue job logs for import errors

2. **Permission issues**:
   - Verify the IAM role has the necessary permissions
   - Check S3 bucket policies

3. **Log format issues**:
   - Ensure logs are in the expected format
   - Adjust the log parsing logic if needed

### Debugging

1. **Check Glue job logs**:
   - Go to the AWS Glue console
   - Find the job run
   - Check the logs for errors

2. **Run the interactive processor**:
   - Use the SageMaker notebook for interactive debugging
   - Add print statements to troubleshoot specific issues

## GitHub Actions Workflows

This project includes several GitHub Actions workflows to automate common tasks:

### 1. Terraform Deploy AWS Infrastructure

The `tf-plan-apply-destroy.yaml` workflow automates Terraform operations:

- **Trigger**: Manual or on push to main branch affecting terraform files
- **Actions**: Plan, Apply, or Destroy infrastructure
- **Usage**: Go to Actions tab, select "Terraform Deploy AWS Infrastructure", and choose the desired action

### 2. Deploy Applications

The `deploy-app.yaml` workflow deploys applications to the EKS cluster:

- **Trigger**: Manual
- **Actions**: Updates kubeconfig, installs Helm, and deploys ArgoCD applications
- **Usage**: Go to Actions tab, select "Deploy Applications", and run the workflow

### 3. Upload Logs to S3

The `upload-logs-to-s3.yml` workflow uploads logs to the S3 bucket:

- **Trigger**: Manual or on push to main branch affecting files in the logs directory
- **Actions**: Uploads logs to the `first-order-application-logs` S3 bucket in the `log-analysis/github-workflow-logs/` prefix
- **Usage**: 
  - Automatic: Push changes to the logs directory
  - Manual: Go to Actions tab, select "Upload Logs to S3", optionally specify custom log path and destination prefix, and run the workflow
- **Storage**: Logs are stored in timestamped directories to preserve history

For more details on the log upload process, see the [logs/README.md](logs/README.md) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
