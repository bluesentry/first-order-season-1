# S3 directories and dependencies setup for Glue job

# Create the necessary S3 directories
resource "aws_s3_object" "glue_temp_dir" {
  bucket       = module.log_bucket.s3_bucket_id
  key          = "glue-temp/"
  content      = ""
  content_type = "application/x-directory"
}

resource "aws_s3_object" "glue_scripts_dir" {
  bucket       = module.log_bucket.s3_bucket_id
  key          = "glue-scripts/"
  content      = ""
  content_type = "application/x-directory"
}

resource "aws_s3_object" "glue_dependencies_dir" {
  bucket       = module.log_bucket.s3_bucket_id
  key          = "glue-dependencies/"
  content      = ""
  content_type = "application/x-directory"
}

resource "aws_s3_object" "log_analysis_dir" {
  bucket       = module.log_bucket.s3_bucket_id
  key          = "log-analysis/"
  content      = ""
  content_type = "application/x-directory"
}
