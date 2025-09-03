# HashiCorp Vault configuration for campus security system

# Storage backend
storage "file" {
  path = "/vault/data"
}

# Listener configuration
listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 1  # Only for development - enable TLS in production
}

# API address
api_addr = "http://0.0.0.0:8200"

# Cluster address
cluster_addr = "http://0.0.0.0:8201"

# UI configuration
ui = true

# Logging
log_level = "INFO"
log_format = "json"

# Disable mlock for containers
disable_mlock = true

# Enable raw endpoint (for health checks)
raw_storage_endpoint = true

# Seal configuration (for production)
# seal "awskms" {
#   region     = "us-west-2"
#   kms_key_id = "alias/vault-unseal-key"
# }

# Plugin directory
plugin_directory = "/vault/plugins"

# Default lease TTL
default_lease_ttl = "768h"  # 32 days
max_lease_ttl = "8760h"     # 1 year