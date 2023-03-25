module "website_s3_bucket" {
  source       = "./modules/google_instance"
  instance_name= "testing_jenkins!"
  project      = "logical-vim-376815"
  region       = "us-central1"
  account_id   = "c-9394"
  display_name = "display_name"
  machine_type = "e2-medium"
  zone         = "us-central1-a"
  image        = "ubuntu-os-cloud/ubuntu-minimal-1804-lts"
  NAME         = "youvilles-ip"
  ADDRESS_TYPE = "EXTERNAL"
  DNS-NAME     = "youville.com."
  MANAGED_ZONE = "devops-villa.com"


}
