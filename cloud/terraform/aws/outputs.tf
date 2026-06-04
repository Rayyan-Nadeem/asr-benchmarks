output "instance_id" {
  value       = aws_instance.this.id
  description = "EC2 instance ID — pass to aws CLI to stop/terminate/describe."
}

output "public_ip" {
  value       = aws_instance.this.public_ip
  description = "Public IP. SSH target."
}

output "public_dns" {
  value       = aws_instance.this.public_dns
  description = "Public DNS name. Use this instead of IP for stable SSH config."
}

output "ssh_command" {
  value       = "ssh -i ${replace(var.ssh_public_key_path, ".pub", "")} ubuntu@${aws_instance.this.public_ip}"
  description = "Copy-paste this to SSH in once the instance finishes booting (~60–90 s)."
}

output "ami_used" {
  value       = data.aws_ami.ubuntu.id
  description = "Which Ubuntu 22.04 AMI was picked. Logged for reproducibility."
}

output "hourly_cost_usd_estimate" {
  value       = "~$0.81/hr for g6.xlarge in us-gov-east-1 (L4 24 GB VRAM). ~$0.53/hr for g4dn.xlarge alternative. Verify in your billing console."
  description = "Reminder of what the meter is running."
}

output "terminate_command" {
  value       = "cd ${path.module} && terraform destroy -auto-approve"
  description = "Run this when measurements are done. The whole instance disappears."
}
