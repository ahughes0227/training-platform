variable "project_id" { type = string }
variable "region" { type = string }
variable "gke_zone" { type = string }
variable "name_prefix" {
  type    = string
  default = "defect-platform"
}
variable "control_image_digest" { type = string }
variable "mlflow_image_digest" { type = string }
variable "gpu_machine_type" { type = string }
variable "gpu_accelerator_type" { type = string }
variable "gpu_count_per_node" {
  type    = number
  default = 1
}
variable "gpu_max_nodes" {
  type    = number
  default = 2
}
variable "db_tier" {
  type    = string
  default = "db-custom-2-7680"
}
variable "db_password" {
  type      = string
  sensitive = true
}
variable "dataset_bucket_name" { type = string }
variable "artifact_bucket_name" { type = string }
variable "class_catalog_bucket_name" { type = string }
variable "catalog_operator_members" {
  type    = list(string)
  default = []
}
variable "infrastructure_capabilities_sha256" {
  type    = string
  default = ""
  validation {
    condition     = var.infrastructure_capabilities_sha256 == "" || can(regex("^[a-f0-9]{64}$", var.infrastructure_capabilities_sha256))
    error_message = "Pin a 64-character lowercase SHA-256 after live infrastructure acceptance."
  }
}
variable "litellm_model" { type = string }
variable "cli_invoker_members" {
  type    = list(string)
  default = []
}
variable "release_operator_members" {
  type    = list(string)
  default = []
  validation {
    condition     = alltrue([for member in var.release_operator_members : startswith(member, "user:")])
    error_message = "release_operator_members must contain Google user:email identities for namespace RBAC."
  }
}
variable "otlp_endpoint" {
  type    = string
  default = ""
}
variable "kuberay_chart_version" {
  type    = string
  default = "1.4.2"
}
