resource "null_resource" "set_github_secrets" {
  triggers = {
    always_run = timestamp() # Forces execution on every apply
  }
  provisioner "local-exec" {
    command = <<EOT
      gh secret set AWS_ACCOUNT_ID --body="$(local.account_id)"
      gh secret set AWS_REGION --body="$(var.region)"
      gh secret set IAM_ROLE_FOR_GITHUB_ACTIONS --body="$(var.github_actions_oidc_role)"
    EOT
  }
  depends_on = [
    aws_iam_role.github_actions_ai_ml_role
  ]
}