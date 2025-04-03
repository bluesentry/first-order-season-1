resource "aws_glue_catalog_database" "glue_db" {
  name = "first-order-glue-db"
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

# IAM Policy for Glue Role to Access S3
resource "aws_iam_role_policy" "glue_policy" {
  role   = aws_iam_role.glue_role.id
  policy = data.aws_iam_policy_document.glue_policy_doc.json
}

data "aws_iam_policy_document" "glue_policy_doc" {
  statement {
    actions = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      "${module.log_bucket.s3_bucket_arn}/*",
      "${module.log_bucket.s3_bucket_arn}"
    ]
  }
}

# Granting Lake Formation Permissions to SSO Admin Role
resource "aws_lakeformation_permissions" "grant_all_to_sso_admin" {
  principal   = "arn:aws:iam::704855531002:role/aws-reserved/sso.amazonaws.com/AWSReservedSSO_AdministratorAccess_de991beb9b0ec0d6"
  permissions = ["ALL"]

  database {
    name = aws_glue_catalog_database.glue_db.name
  }
}

# Granting Lake Formation Permissions to BlueSentry Role
resource "aws_lakeformation_permissions" "grant_all_to_bluesentry" {
  principal   = "arn:aws:iam::704855531002:role/BlueSentry"
  permissions = ["ALL"]

  database {
    name = aws_glue_catalog_database.glue_db.name
  }
}

# Registering S3 Path in Lake Formation for FluentBit Logs
resource "aws_lakeformation_resource" "fluentbit_logs" {
  arn      = "arn:aws:s3:::${module.log_bucket.s3_bucket_id}/fluent-bit-logs"
  role_arn = aws_iam_role.glue_role.arn
}

# Granting Glue Role Permissions to the Glue Database, needed for crawler
resource "aws_lakeformation_permissions" "grant_db_access_to_glue" {
  principal   = aws_iam_role.glue_role.arn
  permissions = ["ALL"]

  database {
    name = aws_glue_catalog_database.glue_db.name
  }
}

# Granting Glue Role Access to the S3 Data Location
resource "aws_lakeformation_permissions" "grant_glue_access_to_fluentbit_logs" {
  principal   = aws_iam_role.glue_role.arn
  permissions = ["DATA_LOCATION_ACCESS"]

  data_location {
    arn = aws_lakeformation_resource.fluentbit_logs.arn
  }
}

# Allowing Glue Role to Decrypt KMS-Encrypted S3 Bucket
resource "aws_iam_role_policy" "glue_kms_policy" {
  role = aws_iam_role.glue_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = "kms:Decrypt",
        Resource = "arn:aws:kms:us-east-1:704855531002:key/YOUR-KMS-KEY-ID"
      }
    ]
  })
}

resource "aws_iam_role_policy" "glue_logs_policy" {
  role = aws_iam_role.glue_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "logs:PutLogEvents",
          "logs:CreateLogStream",
          "logs:CreateLogGroup"
        ],
        Resource = "arn:aws:logs:us-east-1:704855531002:log-group:/aws-glue/crawlers:*"
      }
    ]
  })
}
