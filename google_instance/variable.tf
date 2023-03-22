variable "DNS-NAME" {
type = string
default = "example.com"
description = "dns name"
}

variable "ADDRESS_TYPE" {
type = string 
default = "EXTERNAL"
description = "can be INTERNAL OR EXTERNAL address"  
}


variable "NAME" {
 type = string
 default = "my-static-ip"  
}


variable "project" {
 type = string
 default = "logical-vim-376815"
 description = "project name"
}

variable "region" {
 type = string
 default = "us-central1"
 description = "location of instance"
}

variable "account_id" {
 type = string
 default = "c-1234"
 description = "account identity"
}

variable "display_name" {
 type = string
 default = "c-1234"
 description = "display name"
}

variable "name" {
 type = string
 default = "test"
 description = "name of instance"
}
variable "machine_type" {
 type = string
 default = "e2-medium"
 description = "type of machine used for the instance"
}

variable "zone" {
 type = string
 default = "us-central1-a"
 description = "availability zone for the instance"
}

variable "image" {
 type = string
 default = "ubuntu-os-cloud/ubuntu-minimal-1804-lts"
 description = "image used to spin up the instance"
}
 
 