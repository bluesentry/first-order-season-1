resource "aws_sagemaker_notebook_instance" "sagemaker_notebook" {
  name          = "first-order-LogAnalysisNotebook"
  role_arn      = aws_iam_role.sagemaker_role.arn
  instance_type = "ml.t2.medium"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_iam_role" "sagemaker_role" {
  name = "first-order-aws-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "sagemaker.amazonaws.com"
        },
        Action = "sts:AssumeRole",
      },
    ],
  })
}

resource "aws_iam_role_policy" "sagemaker_s3_access" {
  role = aws_iam_role.sagemaker_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
        ],
        Resource = [
          "${module.log_bucket.s3_bucket_arn}/*",
          "${module.log_bucket.s3_bucket_arn}",
        ],
      },
    ],
  })
}
