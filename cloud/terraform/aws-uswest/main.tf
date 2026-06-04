terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
    # Kept temporarily so we can destroy the DNS record cleanly; will remove
    # in a follow-up commit once the record is gone from state.
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
}

provider "cloudflare" {
  email   = var.cloudflare_email
  api_key = var.cloudflare_global_api_key
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
  default_tags {
    tags = {
      Project = var.project_name
      Owner   = "engineering"
    }
  }
}


# ---------------------------------------------------------------------------
# Networking — use the default VPC + a public subnet from the requested AZ.
# Keeps the surface area small; GovCloud-west default VPC ships with public
# subnets in each AZ.

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_subnet" "first" {
  id = data.aws_subnets.default.ids[0]
}

# ---------------------------------------------------------------------------
# Key pair + security group

resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_key_pair" "this" {
  key_name   = "${var.project_name}-${random_id.suffix.hex}"
  public_key = file(pathexpand(var.ssh_public_key_path))
}

resource "aws_security_group" "this" {
  name        = "${var.project_name}-${random_id.suffix.hex}"
  description = "ASR demo box - SSH from owner IP, HTTP/HTTPS from world"
  vpc_id      = data.aws_vpc.default.id

  # SSH — restrict to the operator's IP only
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # HTTP — used by Let's Encrypt HTTP-01 challenge
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS — the actual demo traffic
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# AMI — Canonical Ubuntu 22.04 LTS via SSM Parameter Store
# (canonical owner ID differs in GovCloud; SSM lookup works in both)

data "aws_ssm_parameter" "ubuntu_ami" {
  name = "/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id"
}

# ---------------------------------------------------------------------------
# Random admin password for the basic-auth gate

resource "random_password" "admin" {
  length  = 24
  special = false # URL-safe; basic auth in browsers handles all chars but
                  # keep it copy-paste-friendly
}

# ---------------------------------------------------------------------------
# EC2 instance

locals {
  user_data = templatefile("${path.module}/user-data.sh.tpl", {
    fqdn           = "${var.subdomain}.${var.zone_name}"
    admin_password = random_password.admin.result
  })
}

resource "aws_instance" "this" {
  ami                         = data.aws_ssm_parameter.ubuntu_ami.value
  instance_type               = var.instance_type
  key_name                    = aws_key_pair.this.key_name
  subnet_id                   = data.aws_subnet.first.id
  vpc_security_group_ids      = [aws_security_group.this.id]
  associate_public_ip_address = true

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens   = "required" # IMDSv2 only
    http_endpoint = "enabled"
  }

  user_data = local.user_data

  tags = {
    Name = var.project_name
  }
}

resource "aws_eip" "this" {
  instance = aws_instance.this.id
  domain   = "vpc"

  tags = {
    Name = "${var.project_name}-eip"
  }
}

# DNS intentionally removed — operator preference: nothing on Cloudflare /
# acmeplexus.com side; access via raw EIP with self-signed TLS only.
