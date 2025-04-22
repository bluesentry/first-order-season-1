# Application Logs

This directory contains application logs that are automatically uploaded to AWS S3 using the GitHub workflow.

## Log Upload Process

Logs in this directory are automatically uploaded to the S3 bucket `first-order-application-logs` in the `log-analysis/github-workflow-logs/` prefix when:

1. Changes are pushed to the main branch that affect files in this directory
2. The workflow is manually triggered from the GitHub Actions tab

## S3 Storage Structure

Logs are stored in S3 with the following structure:
```
first-order-application-logs/
└── log-analysis/
    └── github-workflow-logs/
        └── YYYY-MM-DD-HH-MM-SS/
            ├── application.log
            ├── system_metrics.log
            └── other log files...
```

Each upload creates a new timestamped directory to preserve log history.

## Manual Upload

To manually upload logs:

1. Go to the GitHub repository's Actions tab
2. Select the "Upload Logs to S3" workflow
3. Click "Run workflow"
4. Optionally specify:
   - A custom log path (default: "logs")
   - A custom destination prefix within log-analysis/ (default: "github-workflow-logs")
5. Click "Run workflow"

## Custom Log Paths

When manually triggering the workflow, you can specify a custom log path to upload logs from a different location. This can be:

- A directory path (e.g., "custom-logs/")
- A specific file path (e.g., "specific-file.log")
