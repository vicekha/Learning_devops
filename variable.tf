variable "MANAGED_ZONE" {
type = string 
description = "dns name"
}
variable "DNS-NAME" {
type = string
default = "example.com"
description = "url for dns"
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


variable "Project" {
 type = string
 default = "logical-vim-376815"
 description = "project name"
}

variable "Region" {
 type = string
 default = "us-central1"
 description = "location of instance"
}

variable "Account_id" {
 type = string
 default = "c-1234"
 description = "account identity"
}

variable "Display_name" {
 type = string
 default = "c-1234"
 description = "display name"
}

variable "instance_name" {
 type = string
 default = "test"
 description = "name of instance"
}
variable "Machine_type" {
 type = string
 default = "e2-medium"
 description = "type of machine used for the instance"
}

variable "Zone" {
 type = string
 default = "us-central1-a"
 description = "availability zone for the instance"
}

variable "Image" {
 type = string
 default = "ubuntu-os-cloud/ubuntu-minimal-1804-lts"
 description = "image used to spin up the instance"
}
 
 