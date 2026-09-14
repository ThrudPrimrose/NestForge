#!/usr/bin/env bash
# nest-forge apt setup for an Ubuntu machine (system toolchain only; spack is scripts/setup_spack.sh).
#
# Installs what nest-forge's owned build + arena need from apt: compilers (gcc/clang/gfortran, +flang),
# the OpenMP runtimes (libomp/libgomp), the fast linkers (lld/gold/mold) + LTO archivers
# (gcc-ar/llvm-ar), the vector-math libraries (SLEEF; glibc libmvec is already in libc6), BLAS/LAPACK,
# and the python/build tooling. Vendor toolchains are behind flags (own apt repos + gpg keys):
#   --oneapi  Intel oneAPI  -- icx/icpx/ifx + libiomp5 + SVML + MKL
#   --nvhpc   NVIDIA HPC SDK -- nvc/nvc++/nvfortran + libnvomp
#
# Assumes sudo privileges (uses sudo directly, or runs as-is when already root). Idempotent: present
# packages / repos / gpg keys are detected and skipped. Nothing here is destructive.
#
# Usage: scripts/setup_apt.sh [--oneapi] [--nvhpc] [-h]
#
# Tunables: NVHPC_PKG=nvhpc   (pin e.g. nvhpc-24-3)
set -euo pipefail

NVHPC_PKG="${NVHPC_PKG:-nvhpc}"
DO_ONEAPI=0 DO_NVHPC=0

log()  { printf '\033[1;32m[apt]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[apt:warn]\033[0m %s\n' "$*" >&2; }
# Print the leading comment block (everything after the shebang up to the first non-comment line) as help.
usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next}{exit}' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --oneapi) DO_ONEAPI=1 ;;
    --nvhpc) DO_NVHPC=1 ;;
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

# Install what exists on this release; warn + skip the rest (flang / mold / a libsleef name vary).
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

phase_nvhpc() {
  log "NVIDIA HPC SDK: apt repo + gpg key"
  local key=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
  if [ ! -f "$key" ]; then
    install_gpg_key https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK "$key" \
      || { warn "skipping nvhpc repo (no key)"; return 1; }
    echo "deb [signed-by=$key] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /" \
      | $SUDO tee /etc/apt/sources.list.d/nvhpc.list >/dev/null
    APT_UPDATED=0
  fi
  apt_install "$NVHPC_PKG"
  warn "nvhpc installs under /opt/nvidia/hpc_sdk; add its compilers/bin to PATH (or load its modulefile)."
}

have_ubuntu() { command -v lsb_release >/dev/null 2>&1 && [ "$(lsb_release -is 2>/dev/null)" = "Ubuntu" ]; }
have_ubuntu || warn "not detected as Ubuntu; apt package names may differ"

log "core build + python tooling"
apt_install build-essential cmake ninja-build git pkg-config ca-certificates curl wget gnupg \
            python3 python3-pip python3-venv python3-dev

# ccache: the test suites recompile the same generated kernels repeatedly (per arena cell, per sweep
# rung), and ccache hashes PREPROCESSED SOURCE, not the path, so identical generated C hits the cache
# even from a fresh temp directory. Installing it does NOT enable it -- the shims live in
# /usr/lib/ccache and are used only if that directory is put on PATH. CI deliberately does NOT.
# WARNING 1: never put it on PATH for a job that MEASURES compile time (compare_link_modes,
# perf/staticlib_overhead) -- a cache hit reports ~0s and the
# post-optimization toolchain cost becomes meaningless.
# WARNING 2: never share a cache ACROSS MACHINES while `-march=native` is in the flags. ccache hashes
# that flag as a string and never resolves it to the ISA the compiler chose, so an object built on a
# wider CPU comes back on a narrower one and the kernel dies with SIGILL. This cost CI a red run.
log "compiler cache (ccache): opt-in via PATH; never for compile-time jobs, never shared across CPUs"
apt_install ccache

log "compilers: gcc/g++/gfortran, clang/llvm, flang"
apt_install gcc g++ gfortran clang clang-tools llvm llvm-dev
apt_install flang        # optional: LLVM Fortran, not on every release

log "OpenMP runtimes: libomp (LLVM/clang), libgomp (ships with gcc)"
apt_install libomp-dev libgomp1

log "fast linkers (lld/gold/mold) + LTO archivers (gcc-ar via binutils, llvm-ar via llvm)"
apt_install lld binutils binutils-gold mold

# libsleef-dev ships libsleefgnuabi (the _ZGV* GNU-ABI lib gcc needs) on-path -- no from-source build.
log "vector-math libraries: SLEEF via libsleef-dev (glibc libmvec is part of libc6, already present)"
apt_install libsleef-dev

log "BLAS/LAPACK (arena BLAS axis is a TODO; install the libs now)"
apt_install libopenblas-dev liblapack-dev libblis-dev libfftw3-dev

[ "$DO_ONEAPI" -eq 1 ] && { phase_oneapi || warn "oneAPI setup incomplete"; }
[ "$DO_NVHPC" -eq 1 ] && { phase_nvhpc || warn "nvhpc setup incomplete"; }

log "done. gcc/g++/clang/gfortran on PATH; libomp/libgomp/libsleef + linkers installed."
