data "aws_caller_identity" "current" {}

resource "aws_glue_catalog_database" "glue_db" {
  name = "first-order-glue-db"
}

resource "aws_glue_crawler" "glue_crawler" {
  name          = "first-order-log-crawler"
  role          = "AWSGlueServiceRole"
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

# resource "aws_iam_role_policy" "glue_kms_policy" {
#   role = "AWSGlueServiceRole"  # Reference the built-in Glue role

#   policy = jsonencode({
#     Version = "2012-10-17",
#     Statement = [
#       {
#         Effect   = "Allow",
#         Action   = "kms:Decrypt",
#         Resource = "arn:aws:kms:${var.region}:${data.aws_caller_identity.current.account_id}:key/${data.aws_kms_key.aws_s3.key_id}"
#       }
#     ]
#   })
# }

# resource "aws_iam_role_policy" "glue_logs_policy" {
#   role = "AWSGlueServiceRole"

#   policy = jsonencode({
#     Version = "2012-10-17",
#     Statement = [
#       {
#         Effect = "Allow",
#         Action = [
#           "logs:PutLogEvents",
#           "logs:CreateLogStream",
#           "logs:CreateLogGroup"
#         ],
#         Resource = "arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws-glue/crawlers:*"
#       }
#     ]
#   })
# }

resource "aws_lakeformation_permissions" "grant_all_to_sso_admin" {
  principal   = "arn:aws:iam::704855531002:role/aws-reserved/sso.amazonaws.com/AWSReservedSSO_AdministratorAccess_de991beb9b0ec0d6"
  permissions = ["ALL"]

  database {
    name = aws_glue_catalog_database.glue_db.name
  }
}

resource "aws_lakeformation_permissions" "grant_all_to_bluesentry" {
  principal   = "arn:aws:iam::704855531002:role/BlueSentry"
  permissions = ["ALL"]

  database {
    name = aws_glue_catalog_database.glue_db.name
  }
}

resource "aws_lakeformation_resource" "fluentbit_logs" {
  arn      = "arn:aws:s3:::${module.log_bucket.s3_bucket_id}/fluent-bit-logs"
  role_arn = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/AWSGlueServiceRole"
}

resource "aws_lakeformation_permissions" "grant_db_access_to_glue" {
  principal   = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/AWSGlueServiceRole"
  permissions = ["ALL"]

  database {
    name = aws_glue_catalog_database.glue_db.name
  }
}

resource "aws_lakeformation_permissions" "grant_glue_access_to_fluentbit_logs" {
  principal   = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/AWSGlueServiceRole"
  permissions = ["DATA_LOCATION_ACCESS"]

  data_location {
    arn = aws_lakeformation_resource.fluentbit_logs.arn
  }
}
