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
  arn                     = "arn:aws:s3:::${module.log_bucket.s3_bucket_id}/fluent-bit-logs"
  role_arn                = aws_iam_role.glue_role.arn
  use_service_linked_role = true
}

# Grants Glue role access to the glue db, needed for the crawler to be able to write
resource "aws_lakeformation_permissions" "grant_db_access_to_glue" {
  principal   = aws_iam_role.glue_role.arn
  permissions = ["ALL"]

  database {
    name = aws_glue_catalog_database.glue_db.name
  }
}

# Grants Glue access to the S3 data location
resource "aws_lakeformation_permissions" "grant_glue_access_to_fluentbit_logs" {
  principal   = aws_iam_role.glue_role.arn
  permissions = ["DATA_LOCATION_ACCESS"]

  data_location {
    arn = aws_lakeformation_resource.fluentbit_logs.arn
  }
}
