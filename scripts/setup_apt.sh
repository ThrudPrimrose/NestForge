#!/usr/bin/env bash
# nest-forge apt setup for an Ubuntu machine (system toolchain only; spack is scripts/setup_spack.sh).
#
# Installs what nest-forge's owned build + arena need from apt: compilers (gcc/clang/gfortran, +flang),
# the OpenMP runtimes (libomp/libgomp), BLAS/LAPACK, and the python/build tooling. Intel oneAPI is
# behind a flag (its own apt repo + gpg key):
#   --oneapi  Intel oneAPI  -- icx/icpx/ifx + libiomp5 + SVML + MKL
#
# Assumes sudo privileges (uses sudo directly, or runs as-is when already root). Idempotent: present
# packages / repos / gpg keys are detected and skipped. Nothing here is destructive.
#
# Usage: scripts/setup_apt.sh [--oneapi] [-h]
set -euo pipefail

DO_ONEAPI=0

log()  { printf '\033[1;32m[apt]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[apt:warn]\033[0m %s\n' "$*" >&2; }
# Print the leading comment block (everything after the shebang up to the first non-comment line) as help.
usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next}{exit}' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --oneapi) DO_ONEAPI=1 ;;
    -h|--help) usage 0 ;;
    *) warn "unknown arg: $1"; usage 1 ;;
  esac
  shift
done

# Assume sudo: use it unless we are already root.
SUDO=sudo
[ "$(id -u)" -eq 0 ] && SUDO=""

APT_UPDATED=0
apt_update_once() { [ "$APT_UPDATED" -eq 1 ] || { $SUDO apt-get update -y; APT_UPDATED=1; }; }

# Install what exists on this release; warn + skip the rest (flang's package name varies).
apt_install() {
  apt_update_once
  local ok=() miss=() p
  for p in "$@"; do
    if apt-cache show "$p" >/dev/null 2>&1; then ok+=("$p"); else miss+=("$p"); fi
  done
  [ "${#miss[@]}" -eq 0 ] || warn "not in apt on this release, skipping: ${miss[*]}"
  # DEBIAN_FRONTEND must be set on apt-get's OWN environment: sudo's default env_reset would strip a
  # var exported before it, so pass it THROUGH sudo (via env) -- else debconf can prompt and hang.
  [ "${#ok[@]}" -eq 0 ] || $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${ok[@]}"
}

# Fetch a repo signing key ATOMICALLY: dearmor into a temp file we own, verify it is non-empty, and only
# then install it into place. Guards against a mid-download failure leaving a truncated key that the
# `[ ! -f "$key" ]` guards would treat as "already done" forever. Returns nonzero on failure.
install_gpg_key() {  # <url> <dest-keyring>
  local url="$1" dest="$2" tmp
  tmp=$(mktemp)
  # The $SUDO install is part of the && chain, so ITS failure (read-only /usr, denied sudo) also drops to
  # the failure branch -- the key is only reported installed if it truly landed in $dest.
  if curl -fsSL "$url" | gpg --dearmor >"$tmp" 2>/dev/null && [ -s "$tmp" ] && $SUDO install -m 0644 "$tmp" "$dest"; then
    rm -f "$tmp"; return 0
  fi
  rm -f "$tmp"; warn "failed to fetch/dearmor/install signing key from $url"; return 1
}

phase_oneapi() {
  log "Intel oneAPI: apt repo + gpg key"
  local key=/usr/share/keyrings/oneapi-archive-keyring.gpg
  if [ ! -f "$key" ]; then
    install_gpg_key https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB "$key" \
      || { warn "skipping oneAPI repo (no key)"; return 1; }
    echo "deb [signed-by=$key] https://apt.repos.intel.com/oneapi all main" \
      | $SUDO tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
    APT_UPDATED=0
  fi
  apt_install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-compiler-fortran \
              intel-oneapi-openmp intel-oneapi-mkl intel-oneapi-mkl-devel
  if [ -f /opt/intel/oneapi/setvars.sh ]; then
    log "sourcing oneAPI setvars.sh (puts icx/icpx/ifx + libiomp5/SVML on PATH for the rest of this run)"
    set +u; # shellcheck disable=SC1091
    . /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || warn "setvars.sh returned nonzero"; set -u
  fi
  warn "oneAPI env does NOT persist past this script -- add 'source /opt/intel/oneapi/setvars.sh' to your shell rc."
}

have_ubuntu() { command -v lsb_release >/dev/null 2>&1 && [ "$(lsb_release -is 2>/dev/null)" = "Ubuntu" ]; }
have_ubuntu || warn "not detected as Ubuntu; apt package names may differ"

log "core build + python tooling"
apt_install build-essential cmake ninja-build git pkg-config ca-certificates curl wget gnupg \
            python3 python3-pip python3-venv python3-dev

log "compilers: gcc/g++/gfortran, clang/llvm, flang"
apt_install gcc g++ gfortran clang clang-tools llvm llvm-dev
apt_install flang        # optional: LLVM Fortran, not on every release

log "OpenMP runtimes: libomp (LLVM/clang), libgomp (ships with gcc)"
apt_install libomp-dev libgomp1

log "BLAS/LAPACK"
apt_install libopenblas-dev liblapack-dev libblis-dev libfftw3-dev

[ "$DO_ONEAPI" -eq 1 ] && { phase_oneapi || warn "oneAPI setup incomplete"; }

log "done. gcc/g++/clang/gfortran on PATH; libomp/libgomp installed."
