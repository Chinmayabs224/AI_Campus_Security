#!/bin/bash
# Security hardening script for campus security system deployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# System hardening
harden_system() {
    log "Starting system hardening..."
    
    # Update system packages
    log "Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    
    # Install security tools
    log "Installing security tools..."
    sudo apt install -y \
        fail2ban \
        ufw \
        rkhunter \
        chkrootkit \
        lynis \
        aide \
        auditd \
        apparmor \
        apparmor-utils
    
    # Configure firewall
    log "Configuring UFW firewall..."
    sudo ufw --force reset
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    
    # Allow SSH (adjust port as needed)
    sudo ufw allow 22/tcp
    
    # Allow application ports
    sudo ufw allow 8000/tcp  # API
    sudo ufw allow 3000/tcp  # Frontend
    sudo ufw allow 8200/tcp  # Vault
    sudo ufw allow 5432/tcp  # PostgreSQL (internal)
    sudo ufw allow 6379/tcp  # Redis (internal)
    
    # Enable firewall
    sudo ufw --force enable
    
    success "System hardening completed"
}

# Docker security hardening
harden_docker() {
    log "Hardening Docker configuration..."
    
    # Create Docker daemon configuration
    sudo mkdir -p /etc/docker
    
    cat << 'EOF' | sudo tee /etc/docker/daemon.json
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "live-restore": true,
    "userland-proxy": false,
    "no-new-privileges": true,
    "seccomp-profile": "/etc/docker/seccomp.json",
    "default-ulimits": {
        "nofile": {
            "Name": "nofile",
            "Hard": 64000,
            "Soft": 64000
        }
    },
    "icc": false,
    "userns-remap": "default",
    "disable-legacy-registry": true
}
EOF
    
    # Create Docker seccomp profile
    cat << 'EOF' | sudo tee /etc/docker/seccomp.json
{
    "defaultAction": "SCMP_ACT_ERRNO",
    "archMap": [
        {
            "architecture": "SCMP_ARCH_X86_64",
            "subArchitectures": [
                "SCMP_ARCH_X86",
                "SCMP_ARCH_X32"
            ]
        }
    ],
    "syscalls": [
        {
            "names": [
                "accept",
                "accept4",
                "access",
                "adjtimex",
                "alarm",
                "bind",
                "brk",
                "capget",
                "capset",
                "chdir",
                "chmod",
                "chown",
                "chown32",
                "clock_getres",
                "clock_gettime",
                "clock_nanosleep",
                "close",
                "connect",
                "copy_file_range",
                "creat",
                "dup",
                "dup2",
                "dup3",
                "epoll_create",
                "epoll_create1",
                "epoll_ctl",
                "epoll_ctl_old",
                "epoll_pwait",
                "epoll_wait",
                "epoll_wait_old",
                "eventfd",
                "eventfd2",
                "execve",
                "execveat",
                "exit",
                "exit_group",
                "faccessat",
                "fadvise64",
                "fadvise64_64",
                "fallocate",
                "fanotify_mark",
                "fchdir",
                "fchmod",
                "fchmodat",
                "fchown",
                "fchown32",
                "fchownat",
                "fcntl",
                "fcntl64",
                "fdatasync",
                "fgetxattr",
                "flistxattr",
                "flock",
                "fork",
                "fremovexattr",
                "fsetxattr",
                "fstat",
                "fstat64",
                "fstatat64",
                "fstatfs",
                "fstatfs64",
                "fsync",
                "ftruncate",
                "ftruncate64",
                "futex",
                "getcwd",
                "getdents",
                "getdents64",
                "getegid",
                "getegid32",
                "geteuid",
                "geteuid32",
                "getgid",
                "getgid32",
                "getgroups",
                "getgroups32",
                "getitimer",
                "getpeername",
                "getpgid",
                "getpgrp",
                "getpid",
                "getppid",
                "getpriority",
                "getrandom",
                "getresgid",
                "getresgid32",
                "getresuid",
                "getresuid32",
                "getrlimit",
                "get_robust_list",
                "getrusage",
                "getsid",
                "getsockname",
                "getsockopt",
                "get_thread_area",
                "gettid",
                "gettimeofday",
                "getuid",
                "getuid32",
                "getxattr",
                "inotify_add_watch",
                "inotify_init",
                "inotify_init1",
                "inotify_rm_watch",
                "io_cancel",
                "ioctl",
                "io_destroy",
                "io_getevents",
                "ioprio_get",
                "ioprio_set",
                "io_setup",
                "io_submit",
                "ipc",
                "kill",
                "lchown",
                "lchown32",
                "lgetxattr",
                "link",
                "linkat",
                "listen",
                "listxattr",
                "llistxattr",
                "_llseek",
                "lremovexattr",
                "lseek",
                "lsetxattr",
                "lstat",
                "lstat64",
                "madvise",
                "memfd_create",
                "mincore",
                "mkdir",
                "mkdirat",
                "mknod",
                "mknodat",
                "mlock",
                "mlock2",
                "mlockall",
                "mmap",
                "mmap2",
                "mprotect",
                "mq_getsetattr",
                "mq_notify",
                "mq_open",
                "mq_timedreceive",
                "mq_timedsend",
                "mq_unlink",
                "mremap",
                "msgctl",
                "msgget",
                "msgrcv",
                "msgsnd",
                "msync",
                "munlock",
                "munlockall",
                "munmap",
                "nanosleep",
                "newfstatat",
                "_newselect",
                "open",
                "openat",
                "pause",
                "pipe",
                "pipe2",
                "poll",
                "ppoll",
                "prctl",
                "pread64",
                "preadv",
                "prlimit64",
                "pselect6",
                "ptrace",
                "pwrite64",
                "pwritev",
                "read",
                "readahead",
                "readlink",
                "readlinkat",
                "readv",
                "recv",
                "recvfrom",
                "recvmmsg",
                "recvmsg",
                "remap_file_pages",
                "removexattr",
                "rename",
                "renameat",
                "renameat2",
                "restart_syscall",
                "rmdir",
                "rt_sigaction",
                "rt_sigpending",
                "rt_sigprocmask",
                "rt_sigqueueinfo",
                "rt_sigreturn",
                "rt_sigsuspend",
                "rt_sigtimedwait",
                "rt_tgsigqueueinfo",
                "sched_getaffinity",
                "sched_getattr",
                "sched_getparam",
                "sched_get_priority_max",
                "sched_get_priority_min",
                "sched_getscheduler",
                "sched_rr_get_interval",
                "sched_setaffinity",
                "sched_setattr",
                "sched_setparam",
                "sched_setscheduler",
                "sched_yield",
                "seccomp",
                "select",
                "semctl",
                "semget",
                "semop",
                "semtimedop",
                "send",
                "sendfile",
                "sendfile64",
                "sendmmsg",
                "sendmsg",
                "sendto",
                "setfsgid",
                "setfsgid32",
                "setfsuid",
                "setfsuid32",
                "setgid",
                "setgid32",
                "setgroups",
                "setgroups32",
                "setitimer",
                "setpgid",
                "setpriority",
                "setregid",
                "setregid32",
                "setresgid",
                "setresgid32",
                "setresuid",
                "setresuid32",
                "setreuid",
                "setreuid32",
                "setrlimit",
                "set_robust_list",
                "setsid",
                "setsockopt",
                "set_thread_area",
                "set_tid_address",
                "setuid",
                "setuid32",
                "setxattr",
                "shmat",
                "shmctl",
                "shmdt",
                "shmget",
                "shutdown",
                "sigaltstack",
                "signalfd",
                "signalfd4",
                "sigreturn",
                "socket",
                "socketcall",
                "socketpair",
                "splice",
                "stat",
                "stat64",
                "statfs",
                "statfs64",
                "statx",
                "symlink",
                "symlinkat",
                "sync",
                "sync_file_range",
                "syncfs",
                "sysinfo",
                "tee",
                "tgkill",
                "time",
                "timer_create",
                "timer_delete",
                "timerfd_create",
                "timerfd_gettime",
                "timerfd_settime",
                "timer_getoverrun",
                "timer_gettime",
                "timer_settime",
                "times",
                "tkill",
                "truncate",
                "truncate64",
                "ugetrlimit",
                "umask",
                "uname",
                "unlink",
                "unlinkat",
                "utime",
                "utimensat",
                "utimes",
                "vfork",
                "vmsplice",
                "wait4",
                "waitid",
                "waitpid",
                "write",
                "writev"
            ],
            "action": "SCMP_ACT_ALLOW"
        }
    ]
}
EOF
    
    # Restart Docker daemon
    sudo systemctl restart docker
    
    success "Docker hardening completed"
}

# Configure fail2ban
configure_fail2ban() {
    log "Configuring fail2ban..."
    
    # Create custom jail for campus security
    cat << 'EOF' | sudo tee /etc/fail2ban/jail.d/campus-security.conf
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[campus-api]
enabled = true
port = 8000
filter = campus-api
logpath = /var/log/campus-security/api.log
maxretry = 5
bantime = 1800

[campus-auth]
enabled = true
port = 8000
filter = campus-auth
logpath = /var/log/campus-security/auth.log
maxretry = 3
bantime = 3600
EOF
    
    # Create custom filters
    cat << 'EOF' | sudo tee /etc/fail2ban/filter.d/campus-api.conf
[Definition]
failregex = ^.*\[.*\] ".*" 4\d\d \d+ ".*" ".*" \d+\.\d+\.\d+\.\d+$
ignoreregex =
EOF
    
    cat << 'EOF' | sudo tee /etc/fail2ban/filter.d/campus-auth.conf
[Definition]
failregex = ^.*Authentication failed.*from <HOST>
            ^.*Invalid credentials.*from <HOST>
ignoreregex =
EOF
    
    # Restart fail2ban
    sudo systemctl restart fail2ban
    sudo systemctl enable fail2ban
    
    success "Fail2ban configuration completed"
}

# Set up audit logging
configure_audit() {
    log "Configuring audit logging..."
    
    # Configure auditd rules
    cat << 'EOF' | sudo tee /etc/audit/rules.d/campus-security.rules
# Campus Security System Audit Rules

# Monitor file access
-w /etc/passwd -p wa -k identity
-w /etc/group -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/sudoers -p wa -k identity

# Monitor system calls
-a always,exit -F arch=b64 -S execve -k exec
-a always,exit -F arch=b32 -S execve -k exec

# Monitor network connections
-a always,exit -F arch=b64 -S socket -k network
-a always,exit -F arch=b32 -S socket -k network

# Monitor file system mounts
-a always,exit -F arch=b64 -S mount -k mount
-a always,exit -F arch=b32 -S mount -k mount

# Monitor Docker daemon
-w /usr/bin/docker -p x -k docker
-w /var/lib/docker -p wa -k docker

# Monitor campus security application
-w /app -p wa -k campus-security
-w /var/log/campus-security -p wa -k campus-security-logs

# Make rules immutable
-e 2
EOF
    
    # Restart auditd
    sudo systemctl restart auditd
    sudo systemctl enable auditd
    
    success "Audit logging configuration completed"
}

# Set up log rotation
configure_logrotate() {
    log "Configuring log rotation..."
    
    cat << 'EOF' | sudo tee /etc/logrotate.d/campus-security
/var/log/campus-security/*.log {
    daily
    missingok
    rotate 365
    compress
    delaycompress
    notifempty
    create 0644 www-data www-data
    postrotate
        systemctl reload campus-security-api || true
    endscript
}

/var/log/audit/audit.log {
    daily
    missingok
    rotate 365
    compress
    delaycompress
    notifempty
    create 0600 root root
    postrotate
        systemctl reload auditd || true
    endscript
}
EOF
    
    success "Log rotation configuration completed"
}

# Set up SSL/TLS certificates
setup_ssl() {
    log "Setting up SSL/TLS certificates..."
    
    # Create SSL directory
    sudo mkdir -p /etc/ssl/campus-security
    
    # Generate self-signed certificate for development
    # In production, use proper CA-signed certificates
    sudo openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
        -keyout /etc/ssl/campus-security/private.key \
        -out /etc/ssl/campus-security/certificate.crt \
        -subj "/C=US/ST=State/L=City/O=Campus/OU=Security/CN=campus-security.local"
    
    # Set proper permissions
    sudo chmod 600 /etc/ssl/campus-security/private.key
    sudo chmod 644 /etc/ssl/campus-security/certificate.crt
    
    success "SSL/TLS certificates setup completed"
}

# Create security monitoring script
create_monitoring_script() {
    log "Creating security monitoring script..."
    
    cat << 'EOF' | sudo tee /usr/local/bin/security-monitor.sh
#!/bin/bash
# Campus Security System - Security Monitoring Script

LOG_FILE="/var/log/campus-security/security-monitor.log"
ALERT_EMAIL="security@campus.edu"

# Function to log messages
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check for failed login attempts
check_failed_logins() {
    FAILED_LOGINS=$(grep "authentication failure" /var/log/auth.log | wc -l)
    if [ "$FAILED_LOGINS" -gt 10 ]; then
        log_message "WARNING: High number of failed login attempts: $FAILED_LOGINS"
        echo "High number of failed login attempts detected: $FAILED_LOGINS" | \
            mail -s "Security Alert: Failed Logins" "$ALERT_EMAIL"
    fi
}

# Check disk usage
check_disk_usage() {
    DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 90 ]; then
        log_message "WARNING: High disk usage: ${DISK_USAGE}%"
        echo "High disk usage detected: ${DISK_USAGE}%" | \
            mail -s "Security Alert: Disk Usage" "$ALERT_EMAIL"
    fi
}

# Check for rootkit
check_rootkit() {
    if command -v rkhunter >/dev/null 2>&1; then
        rkhunter --check --skip-keypress --report-warnings-only > /tmp/rkhunter.log 2>&1
        if [ -s /tmp/rkhunter.log ]; then
            log_message "WARNING: Rootkit check found issues"
            cat /tmp/rkhunter.log | mail -s "Security Alert: Rootkit Check" "$ALERT_EMAIL"
        fi
    fi
}

# Check Docker security
check_docker_security() {
    # Check for privileged containers
    PRIVILEGED_CONTAINERS=$(docker ps --filter "label=privileged=true" --format "table {{.Names}}" | tail -n +2)
    if [ -n "$PRIVILEGED_CONTAINERS" ]; then
        log_message "WARNING: Privileged containers running: $PRIVILEGED_CONTAINERS"
    fi
    
    # Check for containers running as root
    ROOT_CONTAINERS=$(docker ps --format "table {{.Names}}\t{{.Command}}" | grep -v "USER" | wc -l)
    if [ "$ROOT_CONTAINERS" -gt 0 ]; then
        log_message "INFO: $ROOT_CONTAINERS containers potentially running as root"
    fi
}

# Main monitoring function
main() {
    log_message "Starting security monitoring check"
    
    check_failed_logins
    check_disk_usage
    check_rootkit
    check_docker_security
    
    log_message "Security monitoring check completed"
}

# Run main function
main
EOF
    
    # Make script executable
    sudo chmod +x /usr/local/bin/security-monitor.sh
    
    # Add to crontab to run every hour
    (crontab -l 2>/dev/null; echo "0 * * * * /usr/local/bin/security-monitor.sh") | crontab -
    
    success "Security monitoring script created"
}

# Main execution
main() {
    log "Starting Campus Security System hardening..."
    
    check_root
    
    # Create log directory
    sudo mkdir -p /var/log/campus-security
    sudo chown www-data:www-data /var/log/campus-security
    
    harden_system
    harden_docker
    configure_fail2ban
    configure_audit
    configure_logrotate
    setup_ssl
    create_monitoring_script
    
    success "Campus Security System hardening completed successfully!"
    
    log "Next steps:"
    log "1. Review and customize firewall rules: sudo ufw status"
    log "2. Check fail2ban status: sudo fail2ban-client status"
    log "3. Review audit logs: sudo ausearch -k campus-security"
    log "4. Test SSL certificates: openssl x509 -in /etc/ssl/campus-security/certificate.crt -text -noout"
    log "5. Monitor security logs: tail -f /var/log/campus-security/security-monitor.log"
}

# Run main function
main "$@"