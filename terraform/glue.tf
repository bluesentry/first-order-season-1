resource "aws_glue_catalog_database" "glue_db" {
  name = "first-order-log-analysis-db"
}

resource "aws_glue_crawler" "glue_crawler" {
  name          = "first-order-log-crawler"
  role          = aws_iam_role.glue_role.arn
  database_name = aws_glue_catalog_database.glue_db.name
  schedule      = "cron(0 12 * * ? *)"

  s3_target {
    path = "s3://${module.log_bucket.s3_bucket_id}/fluent-bit-logs/"
  }

  schema_change_policy {
    delete_behavior = "DELETE_FROM_DATABASE"
    update_behavior = "UPDATE_IN_DATABASE"
  }
}

resource "aws_iam_role" "glue_role" {
  name = "first-order-aws-glue-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "glue.amazonaws.com"
        },
        Action = "sts:AssumeRole",
      },
    ],
  })
}

resource "aws_iam_role_policy" "glue_policy" {
  role   = aws_iam_role.glue_role.id
  policy = data.aws_iam_policy_document.glue_policy_doc.json
}

data "aws_iam_policy_document" "glue_policy_doc" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${module.log_bucket.s3_bucket_arn}/*"]
  }
}
