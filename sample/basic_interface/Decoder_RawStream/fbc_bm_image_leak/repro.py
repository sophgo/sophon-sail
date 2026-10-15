#!/usr/bin/env python3
"""Minimal reproducer for Decoder_RawStream temporary bm_image leakage.

The program is deliberately single-process and single-threaded.  It repeatedly
feeds the same local H.264 Annex-B stream to one Decoder_RawStream and measures
glibc arena bytes that remain allocated.  This removes RTSP, inference,
application frameworks, and concurrency from the reproduction.

Run every comparison arm in a fresh process and against the same input file.
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
import statistics
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter


LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
LIBC.fdopen.restype = ctypes.c_void_p
LIBC.fdopen.argtypes = [ctypes.c_int, ctypes.c_char_p]
LIBC.malloc_info.restype = ctypes.c_int
LIBC.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
LIBC.fflush.argtypes = [ctypes.c_void_p]
LIBC.fclose.argtypes = [ctypes.c_void_p]


def split_annexb_nals(data):
    """Return Annex-B NAL units, including each three/four-byte start code."""
    starts = []
    i = 0
    while i < len(data) - 3:
        if data[i] == 0 and data[i + 1] == 0:
            if data[i + 2] == 1:
                starts.append((i, 3))
                i += 3
                continue
            if data[i + 2] == 0 and data[i + 3] == 1:
                starts.append((i, 4))
                i += 4
                continue
        i += 1

    packets = []
    for index, (start, prefix_len) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(data)
        if start + prefix_len < end:
            nal_type = data[start + prefix_len] & 0x1F
            packets.append((nal_type, data[start:end]))
    return packets


def load_packets(path):
    with open(path, "rb") as stream:
        packets = split_annexb_nals(stream.read())
    if not packets:
        raise RuntimeError("no Annex-B NAL units found in %s" % path)

    # Start at SPS so each run begins from a decoder-initializable point.  Keep
    # SPS/PPS; filtering to picture NALs would make the decoder produce no frame.
    first_sps = next((i for i, item in enumerate(packets) if item[0] == 7), None)
    if first_sps is None:
        raise RuntimeError("no H.264 SPS NAL (type 7) found in %s" % path)
    return [payload for _nal_type, payload in packets[first_sps:]]


def malloc_snapshot():
    """Return per-heap glibc arena accounting without double-counting totals."""
    fd, path = tempfile.mkstemp(prefix="sail-malloc-info-", suffix=".xml")
    try:
        fp = LIBC.fdopen(fd, b"w")
        if not fp:
            os.close(fd)
            raise OSError(ctypes.get_errno(), "fdopen failed")
        fd = -1  # fclose owns it now.
        rc = LIBC.malloc_info(0, ctypes.c_void_p(fp))
        LIBC.fflush(ctypes.c_void_p(fp))
        LIBC.fclose(ctypes.c_void_p(fp))
        if rc != 0:
            raise OSError(ctypes.get_errno(), "malloc_info failed")

        root = ET.parse(path).getroot()
        system_bytes = 0
        free_bytes = 0
        heaps = root.findall("heap")
        for heap in heaps:
            for node in heap.findall("system"):
                if node.get("type") == "current":
                    system_bytes += int(node.get("size", "0"))
            for node in heap.findall("total"):
                if node.get("type") in ("fast", "rest"):
                    free_bytes += int(node.get("size", "0"))

        mmap_bytes = 0
        for node in root.findall("total"):
            if node.get("type") == "mmap":
                mmap_bytes = int(node.get("size", "0"))
        return {
            "heaps": len(heaps),
            "system_bytes": system_bytes,
            "free_bytes": free_bytes,
            "allocated_bytes": system_bytes - free_bytes,
            "mmap_bytes": mmap_bytes,
        }
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def rss_bytes():
    with open("/proc/self/smaps_rollup", "r", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("Rss:"):
                return int(line.split()[1]) * 1024
    return 0


def linear_slope(xs, ys):
    if len(xs) < 3:
        raise RuntimeError("at least three post-warm-up samples are required")
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise RuntimeError("cannot fit a slope to identical x values")
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mib(value):
    return value / 1048576.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h264", required=True, help="raw H.264 Annex-B stream")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--warmup-minutes", type=float, default=10.0)
    parser.add_argument("--sample-seconds", type=float, default=60.0)
    parser.add_argument(
        "--call-rate",
        type=float,
        default=0.0,
        help="read_single_frame calls/s; 0 runs as fast as possible",
    )
    parser.add_argument(
        "--expect",
        choices=("none", "leak", "fixed"),
        default="none",
        help="optional exit-code assertion: leak >200 B/frame; fixed <20 B/frame",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if args.minutes <= args.warmup_minutes:
        parser.error("--minutes must be greater than --warmup-minutes")
    if args.sample_seconds <= 0:
        parser.error("--sample-seconds must be positive")

    from sophon import sail

    packets = load_packets(args.h264)
    sail_path = sail.__file__
    identity = {
        "sail_path": sail_path,
        "sail_sha256": sha256_file(sail_path),
        "target": sail.Handle(args.device).get_target(),
        "h264_path": os.path.abspath(args.h264),
        "h264_sha256": sha256_file(args.h264),
        "packet_count": len(packets),
    }

    print("sail.path      = %s" % identity["sail_path"])
    print("sail.sha256    = %s" % identity["sail_sha256"])
    print("target         = %s" % identity["target"])
    print("h264.sha256    = %s" % identity["h264_sha256"])
    print("packets        = %d" % len(packets))
    print()
    print("  min       calls      frames  ret!=0     RSS MiB    arena system/free/allocated MiB")

    decoder = sail.Decoder_RawStream(args.device, "h264")
    start = time.monotonic()
    stop_at = start + args.minutes * 60.0
    next_sample = start
    next_call = start
    call_interval = 1.0 / args.call_rate if args.call_rate > 0 else 0.0
    calls = 0
    frames = 0
    return_codes = Counter()
    samples = []

    try:
        while time.monotonic() < stop_at:
            now = time.monotonic()
            if call_interval and now < next_call:
                time.sleep(min(next_call - now, 0.002))
                continue

            image = sail.BMImage()
            ret = decoder.read_single_frame(
                packets[calls % len(packets)], image, calls != 0, False
            )
            calls += 1
            return_codes[int(ret)] += 1
            if ret == 0:
                frames += 1
            del image

            if call_interval:
                next_call = max(next_call + call_interval, time.monotonic())

            now = time.monotonic()
            if now >= next_sample:
                snap = malloc_snapshot()
                row = {
                    "elapsed_seconds": now - start,
                    "calls": calls,
                    "frames": frames,
                    "nonzero_returns": calls - frames,
                    "rss_bytes": rss_bytes(),
                    **snap,
                }
                samples.append(row)
                print(
                    "%6.1f  %10d  %10d  %7d  %10.1f    %8.1f/%8.1f/%8.1f"
                    % (
                        row["elapsed_seconds"] / 60.0,
                        calls,
                        frames,
                        row["nonzero_returns"],
                        mib(row["rss_bytes"]),
                        mib(row["system_bytes"]),
                        mib(row["free_bytes"]),
                        mib(row["allocated_bytes"]),
                    ),
                    flush=True,
                )
                next_sample += args.sample_seconds
    finally:
        decoder.release()

    body = [
        row
        for row in samples
        if row["elapsed_seconds"] >= args.warmup_minutes * 60.0
    ]
    frame_x = [row["frames"] for row in body]
    time_x = [row["elapsed_seconds"] / 60.0 for row in body]
    allocated = [row["allocated_bytes"] for row in body]
    rss = [row["rss_bytes"] for row in body]

    result = {
        **identity,
        "minutes": args.minutes,
        "warmup_minutes": args.warmup_minutes,
        "calls": calls,
        "successful_frames": frames,
        "return_codes": dict(sorted(return_codes.items())),
        "sample_count_after_warmup": len(body),
        "allocated_bytes_per_successful_frame": linear_slope(frame_x, allocated),
        "rss_bytes_per_successful_frame": linear_slope(frame_x, rss),
        "allocated_mib_per_minute": linear_slope(time_x, allocated) / 1048576.0,
        "rss_mib_per_minute": linear_slope(time_x, rss) / 1048576.0,
    }

    print()
    print("RESULT")
    print("  successful frames             = %d" % result["successful_frames"])
    print("  return codes                  = %s" % result["return_codes"])
    print(
        "  arena allocated B/frame      = %+.2f"
        % result["allocated_bytes_per_successful_frame"]
    )
    print(
        "  arena allocated MiB/min      = %+.4f"
        % result["allocated_mib_per_minute"]
    )
    print(
        "  RSS B/frame                  = %+.2f"
        % result["rss_bytes_per_successful_frame"]
    )
    print("  RSS MiB/min                  = %+.4f" % result["rss_mib_per_minute"])

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as stream:
            json.dump({"result": result, "samples": samples}, stream, indent=2)
            stream.write("\n")

    value = result["allocated_bytes_per_successful_frame"]
    if args.expect == "leak" and value <= 200.0:
        print("FAIL: expected leak (>200 B/frame), observed %.2f" % value)
        return 2
    if args.expect == "fixed" and value >= 20.0:
        print("FAIL: expected fixed behavior (<20 B/frame), observed %.2f" % value)
        return 2
    if args.expect != "none":
        print("PASS: expectation '%s' satisfied" % args.expect)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
