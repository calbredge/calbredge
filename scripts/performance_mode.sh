#!/bin/bash
# Pin the RK3588's CPU clusters, NPU and memory (DMC) for reproducible benchmarking: the
# RK3588 analogue of jetson_clocks. Every CPU policy, the NPU and the DMC get the performance
# governor, and every CPU policy's scaling maximum is raised to its hardware maximum, which is
# what clocks_pinned requires on this board (docs/RESULT_SCHEMA.md §3.4.1). The GPU is idle
# while the NPU runs a model, so its governor is left alone and only printed.
#
# Orange Pi 5 Plus / RK3588-specific: the NPU and GPU node names are the board's. The settings
# do not survive a reboot (cpufrequtils.service resets the CPU governors at boot); run this
# again before measuring, or install examples/systemd/calbredge-rk3588-clocks.service, which
# runs it at every boot. Run with sudo: writing governors needs root.
#
# It checks every node first and changes nothing unless all of them can be written, so it
# never leaves the board half pinned; a missing node is an error (exit 1), not a skip.
#
# For tests only: CALBREDGE_SYSFS_ROOT (default /) is where sys/ is read and written, and
# CALBREDGE_TEST_EUID overrides the root check.
set -euo pipefail

root=${CALBREDGE_SYSFS_ROOT:-}
cpufreq=$root/sys/devices/system/cpu/cpufreq
npu=$root/sys/class/devfreq/fdab0000.npu
dmc=$root/sys/class/devfreq/dmc
gpu=$root/sys/class/devfreq/fb000000.gpu

fail() {
  echo "performance_mode: $*" >&2
  exit 1
}

[ "${CALBREDGE_TEST_EUID:-$(id -u)}" -eq 0 ] || fail "run with sudo: writing governors needs root"

shopt -s nullglob
policies=("$cpufreq"/policy*/)
shopt -u nullglob
[ "${#policies[@]}" -gt 0 ] || fail "no CPU policies under $cpufreq"

nodes=("$npu/governor" "$dmc/governor")
for policy in "${policies[@]}"; do
  [ -r "${policy}cpuinfo_max_freq" ] || fail "cannot read ${policy}cpuinfo_max_freq"
  nodes+=("${policy}scaling_governor" "${policy}scaling_max_freq")
done
for node in "${nodes[@]}"; do
  [ -w "$node" ] || fail "missing or not writable: $node (nothing was changed)"
done

for policy in "${policies[@]}"; do
  echo performance >"${policy}scaling_governor"
  cat "${policy}cpuinfo_max_freq" >"${policy}scaling_max_freq"
done
echo performance >"$npu/governor"
echo performance >"$dmc/governor"

show() { # a sysfs file's contents, or "unreadable"
  cat "$1" 2>/dev/null || echo unreadable
}

echo "--- state after pinning ---"
for policy in "${policies[@]}"; do
  echo "$(basename "$policy"): $(show "${policy}scaling_governor") @ $(show "${policy}scaling_cur_freq") kHz (max $(show "${policy}scaling_max_freq") of $(show "${policy}cpuinfo_max_freq") kHz)"
done
echo "npu: $(show "$npu/governor") @ $(show "$npu/cur_freq") Hz"
echo "dmc: $(show "$dmc/governor") @ $(show "$dmc/cur_freq") Hz"
echo "gpu: $(show "$gpu/governor") @ $(show "$gpu/cur_freq") Hz (not pinned: idle during NPU inference)"
