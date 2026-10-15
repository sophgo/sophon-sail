# `Decoder_RawStream` temporary `bm_image` leak reproducer

## Summary

`Decoder_RawStream::Decoder_RawStream_CC::fbc_to_bm_image()` creates a
temporary compressed `bm_image` named `fbc_bmimg` for every successfully
decoded FBC frame.  All error paths destroy that image, but the success path in
SAIL v3.11.1 returns without calling `bm_image_destroy(fbc_bmimg)`.

On the tested BM1684X system this leaves approximately **354 bytes of glibc
arena allocation per successfully decoded frame**.  At 25 decoded frames/s,
that is approximately **30 MiB/hour for each decoder process**.  The growth is
unbounded while decoding continues.

This directory contains a one-file, single-process, single-threaded
reproducer.  It does not use RTSP, inference, an application framework, or
concurrent decoders.

## Why a stack variable still needs `bm_image_destroy`

`bm_image fbc_bmimg` is a stack-allocated C descriptor, but
`bm_image_create()` initializes resources owned through that descriptor.  C
does not run a C++ destructor when the descriptor goes out of scope.
`bm_image_destroy()` must therefore release the descriptor's internal
resources.

`bm_image_attach()` attaches device memory supplied by the decoder; it does
not transfer the temporary image descriptor to `out`.  `fbc_bmimg` is only the
input to `bmcv_image_vpp_convert()`, so it remains the caller's responsibility
on both success and failure.

The original source also calls `bm_image_attach()` without assigning its
return value to `ret`; the following check therefore examines the earlier
`bm_image_create()` result.  The proposed change fixes both lifetime and error
propagation and routes all post-create exits through one cleanup point.

## Tested environment

### Target system

| Item | Value |
|---|---|
| Device | BM1684X SoC, device 0 |
| Architecture | aarch64 |
| OS | Kylin Embedded Linux V10 SP1 |
| Kernel | 5.4.217-bm1684 |
| glibc | 2.31 |
| Python | 3.8.10 |
| libsophon | 0.5.1 |
| sophon-ffmpeg | 0.12.0 |
| sophon-opencv | 0.12.0 |
| Installed SAIL version string | `master(20260903_072738)` |
| Installed SAIL `.so` SHA-256 | `fcaa92edc7b654bb256ffbc68895cdf7188edaab591a6d8b831fc2ebb82fd4c2` |

The installed binary and a clean rebuild of the public source showed the same
per-frame growth (difference: 0.43%).

### Cross-build system

| Item | Value |
|---|---|
| Host architecture | x86_64 |
| OS | Ubuntu 20.04.6 LTS |
| Cross compiler | `aarch64-linux-gnu-g++` 9.4.0 |
| CMake | 3.16.3 |
| Python used by the build | 3.8.10 |
| SAIL PR base | v3.11.1, commit `8262effa051f93db8b4bbdfa2fb7d56f5ff43cca` |
| Original `src/cvwrapper.cpp` SHA-256 | `ab78893a111cb454f2b30036b8a4c13d70abbc409082c5b26a3de0a72b3a9367` |

The build used headers and aarch64 libraries from the same target SDK versions
listed above.  Internal hostnames, addresses, credentials, and application
paths are intentionally omitted because they are not required to reproduce
the defect.

## Prepare a local H.264 stream

The test accepts a raw H.264 Annex-B elementary stream.  It contains its own
small Annex-B NAL splitter and has no runtime dependency other than
`sophon.sail`.

Convert any short H.264 MP4 without re-encoding:

```bash
ffmpeg -i input.mp4 -an -c:v copy \
  -bsf:v h264_mp4toannexb -f h264 sample.h264
```

If the input codec is not H.264, create a short H.264 sample:

```bash
ffmpeg -i input.mp4 -t 10 -an -c:v libx264 -preset ultrafast \
  -pix_fmt yuv420p -f h264 sample.h264
```

The stream must include SPS, PPS, and at least one decodable IDR frame.  Use
the exact same `sample.h264` in all A/B runs.

## Run the minimal reproduction

Run each arm in a **fresh process**, because glibc arena high-water state is
process-local.  The script prints the exact imported `.so` path and SHA-256 so
that a system SAIL cannot be mistaken for the candidate build.

Unmodified build:

```bash
PYTHONPATH=/tmp/sail-arms/unpatched/root \
python3 repro.py \
  --h264 sample.h264 \
  --minutes 30 \
  --warmup-minutes 10 \
  --sample-seconds 60 \
  --call-rate 25 \
  --expect leak \
  --json-out unpatched.json
```

Patched build:

```bash
PYTHONPATH=/tmp/sail-arms/patched/root \
python3 repro.py \
  --h264 sample.h264 \
  --minutes 30 \
  --warmup-minutes 10 \
  --sample-seconds 60 \
  --call-rate 25 \
  --expect fixed \
  --json-out patched.json
```

For a shorter smoke test, omit `--call-rate 25` to decode as fast as the device
allows and use `--minutes 10 --warmup-minutes 2 --sample-seconds 30`.

### Measurement semantics

The primary metric is:

```text
arena allocated = sum(per-heap system current)
                - sum(per-heap total fast + total rest)
```

Only `<heap>` nodes are summed.  The root summary is not added again.  In
glibc `malloc_info()`, `fast + rest` describes free chunks, not live objects.
The script fits `arena allocated` directly against the count of **successful
decoded frames** (`ret == 0`) and reports bytes/frame.

RSS is also printed, but it is secondary because glibc can retain freed arena
pages and grow in bounded high-water steps.  `arena allocated` distinguishes
that behavior from live allocation growth.

Default assertions are based on the measured separation:

- `--expect leak`: more than 200 allocated bytes/successful frame;
- `--expect fixed`: fewer than 20 allocated bytes/successful frame.

## Observed results

### This exact `repro.py`

The checked-in script was smoke-tested on the target with the same 10-second
H.264 file in two fresh processes (`50` calls/s, `90` seconds, first `18`
seconds excluded):

| Imported SAIL | Successful frames | Return codes | Arena allocated B/successful frame |
|---|---:|---|---:|
| Installed unmodified SAIL (`fcaa92ed...`) | 4,459 | `{0: 4459, 1: 41}` | **352.92** |
| Final fixed SAIL (`1e71c9d9...`) | 4,459 | `{0: 4459, 1: 41}` | **0.95** |

Both arms decoded the same number of frames and had the same return-code
distribution.  The only changed component was the imported SAIL binary.

### Minimal isolated A/B

One decoder, one thread, 25 calls/s:

| Variant | Arena allocated B/successful frame | Interpretation |
|---|---:|---|
| Unmodified rebuild | 353.98 | Reproduces linear live growth |
| Patched rebuild | 1.98 | Growth removed |
| Installed system SAIL | 352.47 | Matches unmodified rebuild within 0.43% |

### Independent full-path A/B

A separate four-stream validation used the same SAIL-only change in an
`unpatched-patched-patched-unpatched` order.  Values were normalized by
successful decoder frames, not by a lower downstream sampling rate.

| Variant | Arena allocated B/successful frame | Arena allocated MiB/min/worker |
|---|---:|---:|
| Unmodified, median of two arms | 358.20 | +0.5072 |
| Patched, median of two arms | 2.98 | +0.0004 |

The two unmodified arms differed by 0.5%.  The patch reduced per-frame growth
by 99.2% and time-normalized growth by 99.9%.

### Final cleanup implementation

The exact single-cleanup control flow used by this change was rebuilt and
tested separately:

- 25 calls/s for 30 minutes: 44,169 successful frames, zero decode errors,
  0.40 allocated B/frame, +0.0006 MiB/min;
- four streams for 90 minutes: 538,001/538,005 successful calls,
  +0.00567 MiB/min combined long-window allocated slope;
- 12 decoder processes for 22 hours: 23,812,506 successful frames, no OOM,
  no unexpected process loss, and worst final-six-hour allocated slope
  +0.0130 MiB/min.

Some processes showed one-time glibc arena expansion when the heap count
changed.  After each step, `system` and RSS plateaued.  This is allocator
high-water behavior, not the original per-frame live leak.

## How the issue was isolated

The investigation was intentionally reduced from observation to causality:

1. Record the exact imported `sail` path and SHA-256 for every run.
2. Sample one PID over time with `malloc_info()` and `smaps_rollup`.
3. Correct the allocator accounting: per-heap `fast + rest` is free space;
   `system - free` is the approximate live arena allocation.
4. Normalize growth by successful **decoder** frames.  A downstream sampled
   frame rate must not be used when every packet is decoded.
5. Remove RTSP, inference, multiprocessing, and application code, leaving one
   `Decoder_RawStream` and a fixed local packet sequence.
6. Audit create/destroy pairs in the executed source path.  The temporary FBC
   image is created once per successful frame but not destroyed on success.
7. Rebuild two variants from the same source path and change only the SAIL
   cleanup behavior.
8. Run fresh-process A/B tests and then reverse the order to exclude startup
   state and drift.
9. Verify the installed system binary as a third arm.
10. Rebuild the final cleanup implementation, run short functional checks,
    and finish with a multi-process 22-hour soak.

This sequence is included to show how the source observation was connected to
the measured growth; no application-specific code or data is needed.

## Build outline

For BM1684X SoC, point each dependency option at target-matching aarch64
headers and libraries:

```bash
cmake -S /path/to/sophon-sail -B /path/to/sophon-sail/build \
  -DBUILD_TYPE=soc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_SKIP_BUILD_RPATH=ON \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_INSTALL_RPATH= \
  -DLIBSOPHON_BASIC_PATH=/path/to/libsophon-0.5.1 \
  -DFFMPEG_BASIC_PATH=/path/to/sophon-ffmpeg_0.12.0 \
  -DOPENCV_BASIC_PATH=/path/to/sophon-opencv_0.12.0

cmake --build /path/to/sophon-sail/build --target pysail --parallel "$(nproc)"
```

Build the unmodified and patched arms sequentially from the same absolute
source path, cleaning the build directory between arms.  This avoids unrelated
binary differences caused by absolute `__FILE__` strings in logging templates.
Install or unpack the two results into separate private roots; do not overwrite
the system SAIL during the comparison.

## Expected fix behavior

- `fbc_bmimg` is destroyed exactly once after every successful
  `bm_image_create()`.
- A failed `bm_image_create()` returns immediately without destroying an
  uninitialized descriptor.
- The actual return value of `bm_image_attach()` is checked.
- Decode return values, frame rate, and output behavior remain unchanged.
