# Retrieve existing OIDC Provider for GitHub Actions
data "aws_iam_openid_connect_provider" "github_actions" {
  url = "https://token.actions.githubusercontent.com"
}

# IAM Role for GitHub Actions
resource "aws_iam_role" "github_actions_ai_ml_role" {
  name = var.github_actions_oidc_role

  assume_role_policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Effect" : "Allow",
        "Principal" : {
          "Federated" : data.aws_iam_openid_connect_provider.github_actions.arn
        },
        "Action" : "sts:AssumeRoleWithWebIdentity",
        "Condition" : {
          "StringEquals" : {
            "token.actions.githubusercontent.com:aud" : "sts.amazonaws.com"
          },
          "StringLike" : {
            # Adjust the repository identifier to match your GitHub org and repo.
            "token.actions.githubusercontent.com:sub" : "repo:bluesentry/ai_ml_competition_2025_0:*"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "github_actions_workflow" {
  name = "github-actions-workflow-policy"
  role = aws_iam_role.github_actions_ai_ml_role.id

  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Sid" : "ECRPushPermissions",
        "Effect" : "Allow",
        "Action" : [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage", 
          "ecr:GetRepositoryPolicy",
          "ecr:GetLifecyclePolicy"
        ],
        "Resource" : "*"
      },
      {
        "Sid" : "EKSPermissions",
        "Effect" : "Allow",
        "Action" : [
          "eks:CreateCluster",
          "eks:DescribeCluster",
          "eks:UpdateClusterConfig",
          "eks:UpdateClusterVersion",
          "eks:DeleteCluster",
          "eks:ListClusters",
          "eks:DescribeNodegroup",
          "eks:CreateNodegroup",
          "eks:UpdateNodegroupConfig",
          "eks:UpdateNodegroupVersion",
          "eks:DeleteNodegroup"
        ],
        "Resource" : "*"
      },
      {
        "Sid" : "Route53Permissions",
        "Effect" : "Allow",
        "Action" : [
          "route53:ChangeResourceRecordSets",
          "route53:ListResourceRecordSets",
          "route53:GetHostedZone",
          "route53:CreateHostedZone",
          "route53:DeleteHostedZone",
          "route53:ListHostedZones",
          "route53:ListTagsForResource"
        ],
        "Resource" : "*"
      },
      {
        "Sid" : "LoadBalancerPermissions",
        "Effect" : "Allow",
        "Action" : [
          "elasticloadbalancing:CreateLoadBalancer",
          "elasticloadbalancing:DescribeLoadBalancers",
          "elasticloadbalancing:DeleteLoadBalancer",
          "elasticloadbalancing:ModifyLoadBalancerAttributes",
          "elasticloadbalancing:CreateTargetGroup",
          "elasticloadbalancing:DescribeTargetGroups",
          "elasticloadbalancing:DeleteTargetGroup",
          "elasticloadbalancing:RegisterTargets",
          "elasticloadbalancing:DeregisterTargets",
          "elasticloadbalancing:DescribeListeners",
          "elasticloadbalancing:CreateListener",
          "elasticloadbalancing:DeleteListener",
          "elasticloadbalancing:ModifyListener"
        ],
        "Resource" : "*"
      },
      {
        "Sid" : "S3Permissions",
        "Effect" : "Allow",
        "Action" : [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:HeadObject"
        ],
        "Resource" : [
          "arn:aws:s3:::bsc.sandbox.terraform.state",
          "arn:aws:s3:::bsc.sandbox.terraform.state/*"
        ]
      },
      {
        "Sid" : "ECRPermissions",
        "Effect" : "Allow",
        "Action" : [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:CreateRepository",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
          "ecr:DeleteRepository",
          "ecr:DeleteRepositoryPolicy",
          "ecr:SetRepositoryPolicy",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:ListTagsForResource"
        ],
        "Resource" : "*"
      },
      {
        "Sid" : "DynamoDBPermissions",
        "Effect" : "Allow",
        "Action" : [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:Scan",
          "dynamodb:Query",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem",
          "dynamodb:BatchWriteItem",
          "dynamodb:DescribeTable",
          "dynamodb:ListTables"
        ],
        "Resource" : "*"
      },
      {
        "Sid" : "IAMPermissions",
        "Effect" : "Allow",
        "Action" : [
          "iam:PassRole",
          "iam:GetRole",
          "iam:CreateRole",
          "iam:DeleteRole",
          "iam:AttachRolePolicy",
          "iam:DetachRolePolicy",
          "iam:ListAttachedRolePolicies",
          "iam:UpdateAssumeRolePolicy",
          "iam:ListPolicies",
          "iam:GetPolicy",
          "iam:ListOpenIDConnectProviders",
          "iam:ListRolePolicies", 
          "iam:GetPolicy", 
          "iam:GetPolicyVersion",
          "iam:GetOpenIDConnectProvider", 
          "iam:GetRolePolicy"
        ],
        "Resource" : "*"
      },
      {
        "Sid" : "SSMPermissions",
        "Effect" : "Allow",
        "Action" : [
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParameterHistory",
          "ssm:PutParameter",
          "ssm:DeleteParameter",
          "ssm:DescribeParameters"
        ],
        "Resource" : "*"
      },
      {
        "Sid" : "ACMPermissions",
        "Effect" : "Allow",
        "Action" : [
          "acm:DescribeCertificate",
          "acm:ListTagsForCertificate"
        ],
        "Resource" : "*"
      },
      {
        "Sid" : "EC2Permissions",
        "Effect" : "Allow",
        "Action" : [
          "ec2:DescribeVpcs", 
          "ec2:DescribeDhcpOptions",
          "ec2:DescribeAvailabilityZones",
          "ec2:DescribeVpcAttribute",
          "ec2:DescribeSecurityGroups", 
          "ec2:DescribeRouteTables", 
          "ec2:DescribeSubnets", 
          "ec2:DescribeInternetGateways",
          "ec2:DescribeNetworkAcls", 
          "ec2:DescribeFlowLogs",
          "ec2:DescribeSecurityGroupRules",
          "ec2:DescribeAddresses",
          "ec2:DescribeVpcEndpointServices"
        ],
        "Resource" : "*"
      },
      {
        "Sid" : "LogPermissions",
        "Effect" : "Allow",
        "Action" : [
          "logs:DescribeLogGroups",
          "logs:ListTagsForResource"
        ],
        "Resource" : "*"
      },
      {
        "Sid" : "KmsPermissions",
        "Effect" : "Allow",
        "Action" : [
          "kms:DescribeKey", 
          "kms:GetKeyPolicy", 
          "kms:GetKeyRotationStatus"
        ],
        "Resource" : "*"
      },
      {
        "Sid" : "RDSPermissions",
        "Effect" : "Allow",
        "Action" : [
          "rds:DescribeDBSubnetGroups"
        ],
        "Resource" : "*"
      }





    ]
  })
}