resource "google_service_account" "default" {
  account_id   = var.Account_id
  display_name = var.Display_name
}
data "google_dns_managed_zone" "my_managed_zone" {
  name = var.MANAGED_ZONE
  
  
}
resource "google_compute_address" "static_ip" {
  name         = var.NAME
  address_type = var.ADDRESS_TYPE
}

# Create a DNS record that uses the static IP
resource "google_dns_record_set" "dns_record" {
  name    = var.DNS-NAME
  type    = "A"
  ttl     = 300
  rrdatas = [google_compute_address.static_ip.address]
   managed_zone = var.MANAGED_ZONE
}

resource "google_compute_instance" "default" {
  name         = var.instance_name
  machine_type = var.Machine_type
  zone         = var.Zone
  
  provisioner "file" {
    source      = "./Dockerfile"
    destination = "/home"
  }

  boot_disk {
    initialize_params {
      image = var.Image
    }
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.static_ip.address
    }
  }
}

  

  
  
