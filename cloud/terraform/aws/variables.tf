variable "aws_region" {
  description = <<-EOT
    AWS region. Defaults to us-gov-east-1 because the engagement's AWS account is GovCloud. For commercial AWS, switch to us-east-1.
  EOT
  type    = string
  default = "us-gov-east-1"
}

variable "aws_profile" {
  description = "Named AWS profile from ~/.aws/credentials. Empty = use env vars / default chain."
  type        = string
  default     = ""
}

variable "project_name" {
  description = <<-EOT
    Used as prefix for tags + resource names visible in the AWS console.
    Default is intentionally generic so the instance doesn't out the project
    to anyone auditing the AWS account. Override locally via `-var` if you want
    a more descriptive name on a personal AWS account.
  EOT
  type    = string
  default = "eng-gpu-eval"
}

variable "instance_type" {
  description = <<-EOT
    GPU instance type. Defaults to g6.xlarge (L4 24 GB VRAM, 4 vCPU, 16 GB RAM,
    ~$0.81/hr in us-gov-east-1) — Ada Lovelace, succeeds A10G. Available in
    both GovCloud regions (us-gov-east-1, us-gov-west-1).

    Alternatives in GovCloud:
      g4dn.xlarge   — T4 16 GB, ~$0.53/hr, cheaper but Turing (older), still works
    Alternatives in commercial AWS:
      g5.xlarge     — A10G 24 GB, ~$1.01/hr (NOT available in GovCloud)
      g5.2xlarge    — A10G 24 GB, 8 vCPU, 32 GB RAM, ~$1.21/hr
  EOT
  type    = string
  default = "g6.xlarge"
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB. NeMo + Parakeet weights + audio fixtures + Whisper Large v3 = ~25 GB; 100 GB gives plenty of headroom."
  type        = number
  default     = 100
}

variable "ssh_public_key_path" {
  description = "Path to your local public SSH key (uploaded to AWS as a key pair). Generate one with `ssh-keygen -t ed25519 -f ~/.ssh/asr-benchmarks` if you don't have one."
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "allowed_ssh_cidr" {
  description = <<-EOT
    CIDR block allowed to SSH to the instance. STRONGLY recommend restricting to your IP.
    To find your current IP: curl -s https://checkip.amazonaws.com
    Then pass: -var="allowed_ssh_cidr=YOUR.IP.ADDR.HERE/32"
    Default of 0.0.0.0/0 is open-to-internet — fine for a disposable box but a habit not to keep.
  EOT
  type    = string
  default = "0.0.0.0/0"
}
