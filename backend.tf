terraform {
  backend "gcs" {
    bucket = "7dddb4201524b3f9-bucket-tfstate"
    prefix = "terraform/state"
  }
}