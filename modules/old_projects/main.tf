provider "google" {
  project = var.project
  region  = var.region
}
resource "google_service_account" "default" {
  account_id   = var.account_id
  display_name = var.display_name
}
resource "google_compute_address" "static_ip" {
  name         = var.NAME
  address_type = var.ADDRESS_TYPE
}

# Create a DNS record that uses the static IP
resource "google_dns_record_set" "dns_record" {
  name    = "youville.com"
  type    = "A"
  ttl     = 300
  rrdatas = [google_compute_address.static_ip.address]

}
resource "google_compute_instance" "default" {
  name         = var.name
  machine_type = var.machine_type
  zone         = var.zone
  
  provisioner "DockerFile" {
    source = "/Users/devopslearn/Documents/GitHub/Learning_devops/Dockerfile"
    destination = ""
  }

  

  boot_disk {
    initialize_params {
      image = var.image
      
      }
    }
  }

  
  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.static_ip.address
    }
  }

  

  metadata_startup_script = "echo hi > /test.txt"

  service_account {
    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    email  = google_service_account.default.email
    scopes = ["cloud-platform"]
  }



