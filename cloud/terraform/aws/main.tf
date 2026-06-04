# Minimal disposable GPU instance for short-lived engineering evaluation work.
#
# What it provisions:
#   * One EC2 instance (default: g6.xlarge — L4 24 GB VRAM, ~$0.81/hr in us-gov-east-1)
#   * Canonical Ubuntu 22.04 LTS server AMI — bootstrap.sh installs NVIDIA driver
#     + CUDA on top. (DLAMI shortcut documented below for commercial AWS.)
#   * Single security group: SSH from `var.allowed_ssh_cidr` only
#   * Single key pair from your local public SSH key
#   * 100 GB gp3 EBS root volume (ML framework + model weights need ~20 GB; rest is headroom)
#
# What it does NOT provision:
#   * No VPC — uses the default VPC in the chosen region
#   * No persistent storage — destroy means destroy
#   * No IAM role — the instance gets no AWS permissions (we don't need any)
#
# Total cost target: ~$0.81/hr × ~3 hours = $2–4 for one short session.
# Run `terraform destroy` when you're done.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile != "" ? var.aws_profile : null
}

# AMI lookup — Canonical Ubuntu 22.04 LTS amd64 server image.
#
# We default to plain Ubuntu because the engagement's AWS account
# is GovCloud, where the AWS Deep Learning AMI isn't published. bootstrap.sh
# detects the missing NVIDIA driver and installs it (adds ~10–15 min of setup).
#
# ─── SHORTCUT FOR COMMERCIAL AWS (us-east-1, us-west-2, etc.) ────────────────
# If you ever run this on commercial AWS instead of GovCloud, swap the
# `data "aws_ami" "ubuntu"` block below for the DLAMI lookup:
#
#   data "aws_ami" "deep_learning" {
#     most_recent = true
#     owners      = ["898082745236"]  # AWS Deep Learning AMI account
#     filter {
#       name   = "name"
#       values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 22.04)*"]
#     }
#     filter { name = "architecture" values = ["x86_64"] }
#     filter { name = "state"        values = ["available"] }
#   }
#
# Then change `aws_instance.this.ami` to `data.aws_ami.deep_learning.id`.
# That AMI comes with CUDA 12 + NVIDIA driver + PyTorch pre-installed —
# bootstrap.sh's driver-install step (~15 min) becomes a no-op, total
# session wall-clock drops by 15-20 minutes. DLAMI account 898082745236
# is NOT visible from GovCloud, so we can't use it here.
# ─────────────────────────────────────────────────────────────────────────────
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477", "513442679011"] # Canonical (commercial + GovCloud)
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd*/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
  filter {
    name   = "state"
    values = ["available"]
  }
}

# Public key for SSH. Reads from your local ~/.ssh/<keyname>.pub by default.
resource "aws_key_pair" "this" {
  key_name   = "${var.project_name}-${random_id.suffix.hex}"
  public_key = file(var.ssh_public_key_path)

  tags = local.common_tags
}

resource "random_id" "suffix" {
  byte_length = 4
}

# Security group — SSH from the IP range you specify; everything else outbound allowed.
resource "aws_security_group" "this" {
  name        = "${var.project_name}-${random_id.suffix.hex}"
  description = "Disposable engineering eval instance - SSH inbound only"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    description = "All outbound (model downloads, package installs, git push)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# The instance.
resource "aws_instance" "this" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.this.key_name
  vpc_security_group_ids = [aws_security_group.this.id]

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    delete_on_termination = true
    encrypted             = true
  }

  # Tag aggressively so the bill is attributable and the resource is easy to find.
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-${random_id.suffix.hex}"
  })

  # `metadata_options` requires IMDSv2 — AWS security best practice, no functional impact.
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }
}

locals {
  common_tags = {
    ManagedBy   = "terraform"
    AutoDestroy = "true"
  }
}
