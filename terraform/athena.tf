# Create IAM Role for Athena
resource "aws_iam_role" "athena_role" {
  name = "athena_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "athena.amazonaws.com"
        },
        Action = "sts:AssumeRole",
      },
    ],
  })
}

# IAM Policy for Athena Access
resource "aws_iam_policy" "athena_access" {
  name   = "AthenaAccessPolicy"
  policy = data.aws_iam_policy_document.athena_policy.json
}

data "aws_iam_policy_document" "athena_policy" {
  statement {
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
      "s3:GetBucketLocation",
    ]
    resources = [
      "${module.log_bucket.s3_bucket_arn}",
      "${module.log_bucket.s3_bucket_arn}/*",
    ]
  }

  statement {
    actions = [
      "s3:PutObject",
    ]
    resources = [
      "${module.log_bucket.s3_bucket_arn}/athena-results/*",
    ]
  }

  statement {
    actions = [
      "glue:GetDatabase",
      "glue:GetDatabases",
      "glue:GetTable",
      "glue:GetTables",
      "glue:GetPartition",
      "glue:GetPartitions"
    ]
    resources = [
      "*"
    ]
  }
}

# Attach policy to Athena Role
resource "aws_iam_role_policy_attachment" "athena_policy_attachment" {
  role       = aws_iam_role.athena_role.name
  policy_arn = aws_iam_policy.athena_access.arn
}

# Athena Workgroup Configuration
resource "aws_athena_workgroup" "main" {
  name = "main_workgroup"

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true

    result_configuration {
      output_location = "${module.log_bucket.s3_bucket_arn}/athena-results/"
    }
  }

  tags = {
    Purpose = "Log Analysis"
  }
}
