output "public_ip" {
  value       = aws_eip.this.public_ip
  description = "Elastic IP fronting the box."
}

output "ssh_command" {
  value = "ssh -i ${replace(var.ssh_public_key_path, ".pub", "")} ubuntu@${aws_eip.this.public_ip}"
}

output "admin_password" {
  value       = random_password.admin.result
  sensitive   = true
  description = "Run: terraform output -raw admin_password"
}

output "demo_url" {
  value       = "https://${aws_eip.this.public_ip}/"
  description = "Direct IP URL — browser will warn about the self-signed TLS cert (expected)."
}

output "terminate_command" {
  value = "cd ${path.module} && terraform destroy -auto-approve"
}
