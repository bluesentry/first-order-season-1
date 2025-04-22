# Glue Job for Log Summarization
resource "aws_glue_job" "log_summarizer_job" {
  name         = "first-order-log-summarizer"
  role_arn     = aws_iam_role.glue_role.arn
  glue_version = "1.0"
  max_capacity = 1.0 # For Python shell jobs, use max_capacity instead of worker_type
  timeout      = 60  # minutes

  command {
    name            = "pythonshell"
    script_location = "s3://${module.log_bucket.s3_bucket_id}/glue-scripts/log_summarizer_job.py"
    python_version  = "3"
  }

  default_arguments = {
    "--TempDir"                          = "s3://${module.log_bucket.s3_bucket_id}/glue-temp/"
    "--job-language"                     = "python"
    "--enable-continuous-cloudwatch-log" = "true"
    "--enable-metrics"                   = "true"
    "--s3_input_path"                    = "s3://${module.log_bucket.s3_bucket_id}/fluent-bit-logs/"
    "--s3_output_path"                   = "s3://${module.log_bucket.s3_bucket_id}/log-analysis/"
    "--save_model"                       = "true"
    "--extra-py-files"                   = "s3://${module.log_bucket.s3_bucket_id}/glue-dependencies/sentence-transformers-2.2.2-py3-none-any.whl,s3://${module.log_bucket.s3_bucket_id}/glue-dependencies/faiss-1.7.4-py3-none-any.whl"
    "--additional-python-modules"        = "pandas,numpy,torch,sentence-transformers,faiss-cpu"
  }

  execution_property {
    max_concurrent_runs = 1
  }

  tags = var.tags
}

# SNS Topic for Log Analysis Notifications
resource "aws_sns_topic" "log_analysis_notifications" {
  name = "first-order-log-analysis-notifications"
  tags = var.tags
}

# Schedule for the Glue Job (daily run)
resource "aws_glue_trigger" "log_summarizer_daily" {
  name     = "first-order-log-summarizer-daily"
  type     = "SCHEDULED"
  schedule = "cron(0 1 * * ? *)" # Run at 1:00 AM UTC every day

  actions {
    job_name = aws_glue_job.log_summarizer_job.name
    arguments = {
      "--notification_topic" = aws_sns_topic.log_analysis_notifications.arn
    }
  }

  tags = var.tags
}

# S3 bucket policy to allow Glue to access the script
resource "aws_s3_object" "glue_script" {
  bucket = module.log_bucket.s3_bucket_id
  key    = "glue-scripts/log_summarizer_job.py"
  source = "${path.module}/../glue/log_summarizer_job.py"
  etag   = filemd5("${path.module}/../glue/log_summarizer_job.py")
}

# IAM policy for Glue job to access required resources
resource "aws_iam_policy" "glue_log_summarizer_policy" {
  name        = "first-order-glue-log-summarizer-policy"
  description = "Policy for Glue Log Summarizer job"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ],
        Resource = [
          "${module.log_bucket.s3_bucket_arn}",
          "${module.log_bucket.s3_bucket_arn}/*"
        ]
      },
      {
        Effect = "Allow",
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Resource = "arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws-glue/jobs/*"
      },
      {
        Effect = "Allow",
        Action = [
          "glue:GetDatabase",
          "glue:GetTable",
          "glue:GetTables",
          "glue:GetPartition",
          "glue:GetPartitions",
          "glue:BatchGetPartition"
        ],
        Resource = [
          "arn:aws:glue:${var.region}:${data.aws_caller_identity.current.account_id}:catalog",
          "arn:aws:glue:${var.region}:${data.aws_caller_identity.current.account_id}:database/${aws_glue_catalog_database.glue_db.name}",
          "arn:aws:glue:${var.region}:${data.aws_caller_identity.current.account_id}:table/${aws_glue_catalog_database.glue_db.name}/*"
        ]
      },
      {
        Effect = "Allow",
        Action = [
          "sns:Publish"
        ],
        Resource = aws_sns_topic.log_analysis_notifications.arn
      }
    ]
  })
}

# Attach the policy to the Glue role
resource "aws_iam_role_policy_attachment" "glue_log_summarizer_attachment" {
  role       = aws_iam_role.glue_role.name
  policy_arn = aws_iam_policy.glue_log_summarizer_policy.arn
}

# CloudWatch Dashboard for Log Analysis
resource "aws_cloudwatch_dashboard" "log_analysis_dashboard" {
  dashboard_name = "first-order-log-analysis"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "text",
        x      = 0,
        y      = 0,
        width  = 24,
        height = 1,
        properties = {
          markdown = "# Log Analysis Dashboard"
        }
      },
      {
        type   = "metric",
        x      = 0,
        y      = 1,
        width  = 12,
        height = 6,
        properties = {
          metrics = [
            ["AWS/Glue", "glue.driver.aggregate.numCompletedTasks", "JobName", aws_glue_job.log_summarizer_job.name, "JobRunId", "ALL", { "stat" : "Sum" }]
          ],
          view    = "timeSeries",
          stacked = false,
          region  = var.region,
          title   = "Completed Tasks",
          period  = 300
        }
      },
      {
        type   = "metric",
        x      = 12,
        y      = 1,
        width  = 12,
        height = 6,
        properties = {
          metrics = [
            ["AWS/Glue", "glue.driver.aggregate.numFailedTasks", "JobName", aws_glue_job.log_summarizer_job.name, "JobRunId", "ALL", { "stat" : "Sum" }]
          ],
          view    = "timeSeries",
          stacked = false,
          region  = var.region,
          title   = "Failed Tasks",
          period  = 300
        }
      },
      {
        type   = "metric",
        x      = 0,
        y      = 7,
        width  = 24,
        height = 6,
        properties = {
          metrics = [
            ["AWS/Glue", "glue.driver.aggregate.elapsedTime", "JobName", aws_glue_job.log_summarizer_job.name, "JobRunId", "ALL", { "stat" : "Average" }]
          ],
          view    = "timeSeries",
          stacked = false,
          region  = var.region,
          title   = "Job Execution Time (ms)",
          period  = 300
        }
      }
    ]
  })
}

# Output the SNS topic ARN for subscribing to notifications
output "log_analysis_sns_topic_arn" {
  description = "ARN of the SNS topic for log analysis notifications"
  value       = aws_sns_topic.log_analysis_notifications.arn
}

# Output the Glue job name
output "log_summarizer_job_name" {
  description = "Name of the Glue job for log summarization"
  value       = aws_glue_job.log_summarizer_job.name
}

# Output the CloudWatch dashboard URL
output "log_analysis_dashboard_url" {
  description = "URL of the CloudWatch dashboard for log analysis"
  value       = "https://${var.region}.console.aws.amazon.com/cloudwatch/home?region=${var.region}#dashboards:name=${aws_cloudwatch_dashboard.log_analysis_dashboard.dashboard_name}"
}
