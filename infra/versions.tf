terraform {
  required_version = ">= 1.8.0"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 6.0" }
    helm   = { source = "hashicorp/helm", version = "~> 2.17" }
    kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.36" }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_client_config" "current" {}

provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.ray.endpoint}"
    token                  = data.google_client_config.current.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.ray.master_auth[0].cluster_ca_certificate)
  }
}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.ray.endpoint}"
  token                  = data.google_client_config.current.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.ray.master_auth[0].cluster_ca_certificate)
}
