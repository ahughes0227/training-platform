locals {
  apis = toset([
    "aiplatform.googleapis.com", "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com", "compute.googleapis.com",
    "container.googleapis.com", "run.googleapis.com",
    "sqladmin.googleapis.com", "secretmanager.googleapis.com",
    "storage.googleapis.com", "workflows.googleapis.com",
    "cloudtasks.googleapis.com", "logging.googleapis.com",
  ])
}

resource "google_project_service" "required" {
  for_each           = local.apis
  service            = each.key
  disable_on_destroy = false
}

resource "google_storage_bucket" "datasets" {
  name                        = var.dataset_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  versioning { enabled = true }
  depends_on = [google_project_service.required]
}

resource "google_storage_bucket" "artifacts" {
  name                        = var.artifact_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  versioning { enabled = true }
  depends_on = [google_project_service.required]
}

# Catalog approval authority is separated from dataset/trainer write authority.
resource "google_storage_bucket" "catalogs" {
  name                        = var.class_catalog_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  versioning { enabled = true }
  depends_on = [google_project_service.required]
}

resource "google_storage_bucket_iam_member" "catalog_readers" {
  for_each = {
    control = google_service_account.control.email
    trainer = google_service_account.trainer.email
    serve   = google_service_account.serve.email
    dataset = google_service_account.dataset.email
  }
  bucket = google_storage_bucket.catalogs.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${each.value}"
}

resource "google_storage_bucket_iam_member" "catalog_operator_creates" {
  for_each = toset(var.catalog_operator_members)
  bucket   = google_storage_bucket.catalogs.name
  role     = "roles/storage.objectCreator"
  member   = each.value
}

resource "google_storage_bucket_iam_member" "catalog_operator_reads" {
  for_each = toset(var.catalog_operator_members)
  bucket   = google_storage_bucket.catalogs.name
  role     = "roles/storage.objectViewer"
  member   = each.value
}

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = "${var.name_prefix}-images"
  format        = "DOCKER"
  depends_on    = [google_project_service.required]
}

resource "google_service_account" "control" {
  account_id   = "defect-control"
  display_name = "Defect platform controller"
}

resource "google_service_account" "trainer" {
  account_id   = "defect-trainer"
  display_name = "Defect Vertex trainer"
}

resource "google_service_account" "mlflow" {
  account_id   = "defect-mlflow"
  display_name = "Defect MLflow server"
}

resource "google_service_account" "serve" {
  account_id   = "defect-serving"
  display_name = "Defect Ray Serve"
}

resource "google_service_account" "workflow" {
  account_id   = "defect-workflow"
  display_name = "Defect durable workflow"
}

resource "google_service_account" "dataset" {
  account_id   = "defect-dataset"
  display_name = "Defect dataset publisher"
}

resource "google_service_account" "runtime" {
  account_id   = "defect-runtime"
  display_name = "Defect runtime image builder"
}

resource "google_service_account" "certifier" {
  account_id   = "defect-certifier"
  display_name = "Defect Vertex runtime certifier"
}

resource "google_storage_bucket_iam_member" "trainer_datasets" {
  bucket = google_storage_bucket.datasets.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.trainer.email}"
}

resource "google_storage_bucket_iam_member" "control_datasets" {
  bucket = google_storage_bucket.datasets.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.control.email}"
}

resource "google_storage_bucket_iam_member" "dataset_publisher" {
  bucket = google_storage_bucket.datasets.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.dataset.email}"
}

resource "google_storage_bucket_iam_member" "dataset_publisher_reads" {
  bucket = google_storage_bucket.datasets.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.dataset.email}"
}

resource "google_storage_bucket_iam_member" "trainer_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.trainer.email}"
}

resource "google_storage_bucket_iam_member" "control_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.control.email}"
}

resource "google_storage_bucket_iam_member" "certifier_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.certifier.email}"
}

resource "google_artifact_registry_repository_iam_member" "runtime_image_writer" {
  project    = var.project_id
  location   = google_artifact_registry_repository.images.location
  repository = google_artifact_registry_repository.images.repository_id
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_storage_bucket_iam_member" "mlflow_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.mlflow.email}"
}

resource "google_storage_bucket_iam_member" "serve_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.serve.email}"
}

resource "google_storage_bucket_iam_member" "release_operator_creates_ledger" {
  for_each = toset(var.release_operator_members)
  bucket   = google_storage_bucket.artifacts.name
  role     = "roles/storage.objectCreator"
  member   = each.value
}

resource "google_storage_bucket_iam_member" "release_operator_reads_ledger" {
  for_each = toset(var.release_operator_members)
  bucket   = google_storage_bucket.artifacts.name
  role     = "roles/storage.objectViewer"
  member   = each.value
}

resource "google_project_iam_member" "control_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.control.email}"
}

resource "google_project_iam_member" "certifier_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.certifier.email}"
}

resource "google_project_iam_member" "workflow_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.viewer"
  member  = "serviceAccount:${google_service_account.workflow.email}"
}

resource "google_project_iam_member" "control_workflow" {
  project = var.project_id
  role    = "roles/workflows.invoker"
  member  = "serviceAccount:${google_service_account.control.email}"
}

resource "google_service_account_iam_member" "control_can_use_trainer" {
  service_account_id = google_service_account.trainer.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.control.email}"
}

resource "google_service_account_iam_member" "certifier_can_use_trainer" {
  service_account_id = google_service_account.trainer.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.certifier.email}"
}

resource "google_sql_database_instance" "state" {
  name                = "${var.name_prefix}-state"
  region              = var.region
  database_version    = "POSTGRES_16"
  deletion_protection = true
  settings {
    tier              = var.db_tier
    availability_type = "REGIONAL"
    backup_configuration { enabled = true }
    ip_configuration {
      ipv4_enabled = true
      ssl_mode     = "ENCRYPTED_ONLY"
    }
  }
  depends_on = [google_project_service.required]
}

resource "google_sql_database" "platform" {
  name     = "platform"
  instance = google_sql_database_instance.state.name
}

resource "google_sql_database" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.state.name
}

resource "google_sql_user" "platform" {
  name     = "platform"
  instance = google_sql_database_instance.state.name
  password = var.db_password
}

resource "google_sql_user" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.state.name
  password = var.db_password
}

resource "google_secret_manager_secret" "db_password" {
  secret_id = "${var.name_prefix}-db-password"
  replication {
    auto {}
  }
  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}

resource "google_secret_manager_secret_iam_member" "control_db" {
  secret_id = google_secret_manager_secret.db_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.control.email}"
}

resource "google_secret_manager_secret_iam_member" "mlflow_db" {
  secret_id = google_secret_manager_secret.db_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.mlflow.email}"
}

resource "google_project_iam_member" "control_sql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.control.email}"
}

resource "google_project_iam_member" "mlflow_sql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.mlflow.email}"
}

resource "google_cloud_run_v2_service" "mlflow" {
  name     = "${var.name_prefix}-mlflow"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"
  template {
    service_account = google_service_account.mlflow.email
    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
    volumes {
      name = "cloudsql"
      cloud_sql_instance { instances = [google_sql_database_instance.state.connection_name] }
    }
    containers {
      image = var.mlflow_image_digest
      ports { container_port = 8080 }
      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
      env {
        name  = "MLFLOW_ARTIFACTS_DESTINATION"
        value = "gs://${google_storage_bucket.artifacts.name}/mlflow"
      }
      env {
        name  = "DB_NAME"
        value = google_sql_database.mlflow.name
      }
      env {
        name  = "DB_USER"
        value = google_sql_user.mlflow.name
      }
      env {
        name  = "DB_SOCKET"
        value = "/cloudsql/${google_sql_database_instance.state.connection_name}"
      }
      env {
        name = "DB_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_password.secret_id
            version = "latest"
          }
        }
      }
    }
  }
  depends_on = [google_project_service.required]
}

resource "google_cloud_run_v2_service" "control" {
  name     = "${var.name_prefix}-control"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"
  template {
    service_account = google_service_account.control.email
    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }
    volumes {
      name = "cloudsql"
      cloud_sql_instance { instances = [google_sql_database_instance.state.connection_name] }
    }
    containers {
      image = var.control_image_digest
      ports { container_port = 8080 }
      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
      env {
        name  = "DEFECT_GCP_PROJECT"
        value = var.project_id
      }
      env {
        name  = "DEFECT_CLASS_CATALOG_ROOT"
        value = "gs://${google_storage_bucket.catalogs.name}/approved"
      }
      env {
        name  = "DEFECT_INFRA_CAPABILITIES_FILE"
        value = var.infrastructure_capabilities_sha256 == "" ? "" : "gs://${google_storage_bucket.catalogs.name}/infrastructure/${var.infrastructure_capabilities_sha256}.json"
      }
      env {
        name  = "DEFECT_INFRA_CAPABILITIES_SHA256"
        value = var.infrastructure_capabilities_sha256
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "DEFECT_GCP_REGION"
        value = var.region
      }
      env {
        name  = "GOOGLE_CLOUD_REGION"
        value = var.region
      }
      env {
        name  = "DEFECT_WORKFLOW_NAME"
        value = "${var.name_prefix}-train"
      }
      env {
        name  = "DEFECT_MLFLOW_IAM_AUTH"
        value = "1"
      }
      env {
        name  = "DEFECT_LITELLM_MODEL"
        value = var.litellm_model
      }
      env {
        name  = "DEFECT_OTLP_ENDPOINT"
        value = var.otlp_endpoint
      }
      env {
        name  = "DEFECT_MLFLOW_TRACKING_URI"
        value = google_cloud_run_v2_service.mlflow.uri
      }
      env {
        name  = "DEFECT_STATE_DATABASE"
        value = google_sql_database.platform.name
      }
      env {
        name  = "DEFECT_STATE_DB_USER"
        value = google_sql_user.platform.name
      }
      env {
        name  = "DEFECT_STATE_DB_SOCKET"
        value = "/cloudsql/${google_sql_database_instance.state.connection_name}"
      }
      env {
        name = "DEFECT_STATE_DB_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_password.secret_id
            version = "latest"
          }
        }
      }
    }
  }
  depends_on = [google_project_service.required]
}

resource "google_cloud_run_v2_service_iam_member" "control_invokes_mlflow" {
  project  = var.project_id
  location = google_cloud_run_v2_service.mlflow.location
  name     = google_cloud_run_v2_service.mlflow.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.control.email}"
}

resource "google_cloud_run_v2_service_iam_member" "trainer_invokes_mlflow" {
  project  = var.project_id
  location = google_cloud_run_v2_service.mlflow.location
  name     = google_cloud_run_v2_service.mlflow.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.trainer.email}"
}

resource "google_cloud_run_v2_service_iam_member" "serve_invokes_mlflow" {
  project  = var.project_id
  location = google_cloud_run_v2_service.mlflow.location
  name     = google_cloud_run_v2_service.mlflow.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.serve.email}"
}

resource "google_cloud_run_v2_service_iam_member" "release_operator_invokes_mlflow" {
  for_each = toset(var.release_operator_members)
  project  = var.project_id
  location = google_cloud_run_v2_service.mlflow.location
  name     = google_cloud_run_v2_service.mlflow.name
  role     = "roles/run.invoker"
  member   = each.value
}

resource "google_project_iam_member" "release_operator_views_cluster" {
  for_each = toset(var.release_operator_members)
  project  = var.project_id
  role     = "roles/container.clusterViewer"
  member   = each.value
}

resource "google_service_account_iam_member" "ray_workload_identity" {
  service_account_id = google_service_account.serve.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[defect-serving/defect-serving]"
}

resource "google_cloud_run_v2_service_iam_member" "workflow_invokes_control" {
  project  = var.project_id
  location = google_cloud_run_v2_service.control.location
  name     = google_cloud_run_v2_service.control.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.workflow.email}"
}

resource "google_cloud_run_v2_service_iam_member" "cli_invokes_control" {
  for_each = toset(var.cli_invoker_members)
  project  = var.project_id
  location = google_cloud_run_v2_service.control.location
  name     = google_cloud_run_v2_service.control.name
  role     = "roles/run.invoker"
  member   = each.value
}

resource "google_workflows_workflow" "train" {
  name            = "${var.name_prefix}-train"
  region          = var.region
  service_account = google_service_account.workflow.email
  user_env_vars   = { CONTROL_URL = google_cloud_run_v2_service.control.uri }
  source_contents = file("${path.module}/workflow.yaml")
  depends_on      = [google_project_service.required]
}

resource "google_cloud_tasks_queue" "short_tasks" {
  name     = "${var.name_prefix}-short-tasks"
  location = var.region
  rate_limits { max_dispatches_per_second = 5 }
  retry_config { max_attempts = 5 }
  depends_on = [google_project_service.required]
}

resource "google_container_cluster" "ray" {
  name                     = "${var.name_prefix}-ray"
  location                 = var.gke_zone
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = true
  workload_identity_config { workload_pool = "${var.project_id}.svc.id.goog" }
  depends_on = [google_project_service.required]
}

resource "google_container_node_pool" "system" {
  name       = "system"
  location   = var.gke_zone
  cluster    = google_container_cluster.ray.name
  node_count = 1
  node_config {
    machine_type = "e2-standard-4"
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    workload_metadata_config { mode = "GKE_METADATA" }
  }
}

resource "google_container_node_pool" "gpu" {
  name     = "gpu-serving"
  location = var.gke_zone
  cluster  = google_container_cluster.ray.name
  autoscaling {
    min_node_count = 0
    max_node_count = var.gpu_max_nodes
  }
  node_config {
    machine_type = var.gpu_machine_type
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    guest_accelerator {
      type  = var.gpu_accelerator_type
      count = var.gpu_count_per_node
    }
    labels = { workload = "ray-serve" }
    workload_metadata_config { mode = "GKE_METADATA" }
  }
}

resource "helm_release" "kuberay_operator" {
  name             = "kuberay-operator"
  repository       = "https://ray-project.github.io/kuberay-helm/"
  chart            = "kuberay-operator"
  version          = var.kuberay_chart_version
  namespace        = "kuberay-system"
  create_namespace = true
  depends_on       = [google_container_node_pool.system]
}

resource "kubernetes_namespace_v1" "serving" {
  metadata { name = "defect-serving" }
  depends_on = [google_container_node_pool.system]
}

resource "kubernetes_service_account_v1" "serving" {
  metadata {
    name      = "defect-serving"
    namespace = kubernetes_namespace_v1.serving.metadata[0].name
    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.serve.email
    }
  }
}

resource "kubernetes_role_v1" "release_operator" {
  metadata {
    name      = "defect-release-operator"
    namespace = kubernetes_namespace_v1.serving.metadata[0].name
  }
  rule {
    api_groups = ["ray.io"]
    resources  = ["rayservices"]
    verbs      = ["get", "list", "watch", "create", "update", "patch"]
  }
}

resource "kubernetes_role_binding_v1" "release_operator" {
  for_each = toset(var.release_operator_members)
  metadata {
    name      = "defect-release-${substr(sha256(each.value), 0, 12)}"
    namespace = kubernetes_namespace_v1.serving.metadata[0].name
  }
  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "Role"
    name      = kubernetes_role_v1.release_operator.metadata[0].name
  }
  subject {
    kind      = "User"
    name      = trimprefix(each.value, "user:")
    api_group = "rbac.authorization.k8s.io"
  }
}
