module "website_s3_bucket" {
  source       = "./modules/google_instance2.0"
  instance_name= "testing_jenkins!"
  Project      = "logical-vim-376815"
  Region       = "us-central1"
  Account_id   = "c-9394"
  Display_name = "display_name"
  Machine_type = "e2-medium"
  Zone         = "us-central1-a"
  Image        = "ubuntu-os-cloud/ubuntu-minimal-1804-lts"
  NAME         = "youvilles-ip"
  ADDRESS_TYPE = "EXTERNAL"
  DNS-NAME     = "youville.com."
  MANAGED_ZONE = "devops-villa.com"


}
