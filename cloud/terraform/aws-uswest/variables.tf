variable "aws_region" {
  description = "AWS region. GovCloud west since the synergy-nautilus profile is GovCloud."
  type        = string
  default     = "us-gov-west-1"
}

variable "aws_profile" {
  description = "Named profile from ~/.aws/credentials."
  type        = string
  default     = "synergy-nautilus"
}

variable "project_name" {
  description = "Generic tag prefix; nothing client-identifying on the surface."
  type        = string
  default     = "model-playground"
}

variable "instance_type" {
  description = <<-EOT
    GPU instance shape. g4dn.xlarge = Tesla T4 16 GB (sm_75 / Turing — same
    architecture class as RTX 2060/2070). Cheap consumer-equivalent GPU.
    ~$0.526/hr in us-gov-west-1 commercial; GovCloud rates may differ.
    Verify in the billing console after the first hour.
  EOT
  type        = string
  default     = "g4dn.xlarge"
}

variable "root_volume_gb" {
  description = "Root volume; needs headroom for NeMo + Parakeet + nemotron-streaming weights + Docker images."
  type        = number
  default     = 80
}

variable "ssh_public_key_path" {
  description = "Local path to the SSH public key uploaded to AWS."
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed for inbound SSH. Tighten to YOUR.IP/32 — get via curl -s https://checkip.amazonaws.com"
  type        = string
}

# ---------------------------------------------------------------------------
# DNS

variable "zone_name" {
  description = "Cloudflare zone (the apex domain)."
  type        = string
  default     = "acmeplexus.com"
}

variable "subdomain" {
  description = "Subdomain label, joined to zone_name with a dot."
  type        = string
  default     = "model-playground"
}

variable "cloudflare_email" {
  description = "Cloudflare account email for Global API Key auth."
  type        = string
}

variable "cloudflare_global_api_key" {
  description = <<-EOT
    Cloudflare Global API Key. Pass via TF_VAR_cloudflare_global_api_key env
    var — never via tfvars in git. Source of truth: ~/.aws/credentials
    cloudflare-acmeplexus section.
  EOT
  type        = string
  sensitive   = true
}
