# Validation results

All measurements below were made on the same BM1684X target described in
`README.md`.  Each arm ran in a fresh process and printed the imported SAIL
path and SHA-256 before decoding.

## 1. Checked-in minimal reproducer

Test shape:

- one process, one thread, one `Decoder_RawStream`;
- the same local 1280x720 H.264 Annex-B file in both arms;
- file SHA-256:
  `2b833a125703cafeca8139acd50f171e32420942aaa5e70a4ff1aa4d6cfe441a`;
- 50 `read_single_frame()` calls/s for 90 seconds;
- 15-second memory samples;
- first 18 seconds excluded from regression.

### Unmodified installed SAIL

```text
sail.sha256 = fcaa92edc7b654bb256ffbc68895cdf7188edaab591a6d8b831fc2ebb82fd4c2
target      = BM1684X
packets     = 243

  min       calls      frames  ret!=0     RSS MiB    arena system/free/allocated MiB
   0.0           1           0        1        50.1        10.0/     1.7/     8.2
   0.3         751         740       11        51.0        10.4/     1.8/     8.5
   0.5        1501        1484       17        51.0        10.4/     1.6/     8.8
   0.8        2251        2228       23        51.0        10.4/     1.3/     9.0
   1.0        3001        2972       29        51.0        10.4/     1.1/     9.3
   1.3        3751        3716       35        51.0        10.4/     0.8/     9.5

RESULT
  successful frames             = 4459
  return codes                  = {0: 4459, 1: 41}
  arena allocated B/frame      = +352.92
  arena allocated MiB/min      = +1.0016
  RSS B/frame                  = +1.65
  RSS MiB/min                  = +0.0047
PASS: expectation 'leak' satisfied
```

The short-window RSS slope is nearly flat because glibc first consumes free
chunks already held in its arena.  `system` is stable while `free` falls and
`allocated = system - free` rises by 352.92 bytes/frame.  This is exactly why
RSS alone is not the primary short-reproduction metric.

### Final fixed SAIL

```text
sail.sha256 = 1e71c9d9d5718f7109c52b073442ac54b2e94a04a4010405f02947a35b51fb8c
target      = BM1684X
packets     = 243

  min       calls      frames  ret!=0     RSS MiB    arena system/free/allocated MiB
   0.0           1           0        1        50.2        10.0/     1.7/     8.2
   0.3         751         740       11        51.1        10.4/     2.1/     8.3
   0.5        1501        1484       17        51.1        10.4/     2.1/     8.3
   0.8        2251        2228       23        51.1        10.4/     2.1/     8.3
   1.0        3001        2972       29        51.1        10.4/     2.1/     8.3
   1.3        3751        3716       35        51.1        10.4/     2.1/     8.3

RESULT
  successful frames             = 4459
  return codes                  = {0: 4459, 1: 41}
  arena allocated B/frame      = +0.95
  arena allocated MiB/min      = +0.0027
  RSS B/frame                  = +1.65
  RSS MiB/min                  = +0.0047
PASS: expectation 'fixed' satisfied
```

The frame count and return-code distribution are identical.  Arena allocated
bytes/frame fell from 352.92 to 0.95 (99.7%).

## 2. Longer isolated A/B and installed-binary cross-check

| Arm | Arena allocated B/successful frame |
|---|---:|
| Clean unmodified rebuild | 353.98 |
| Installed system SAIL | 352.47 |
| Rebuild with success-path destroy | 1.98 |

The installed SAIL differed from the clean unmodified rebuild by only 0.43%,
showing that the source-level A/B result applies to the binary originally
observed on the target.

## 3. Full-path reverse-order A/B

Four independent decoder streams were run in the order
unmodified-fixed-fixed-unmodified.  Only the SAIL binary changed.

| Arm | Arena allocated B/successful frame | Arena allocated MiB/min/worker |
|---|---:|---:|
| Unmodified arm 1 | 359.07 | +0.5035 |
| Fixed arm 1 | 5.64 | +0.0041 |
| Fixed arm 2 | 0.33 | -0.0033 |
| Unmodified arm 2 | 357.33 | +0.5109 |
| Unmodified median | **358.20** | **+0.5072** |
| Fixed median | **2.98** | **+0.0004** |

The two unmodified arms differ by 0.5%.  The fixed arms eliminate 99.2% of
per-frame allocated growth and 99.9% of time-normalized growth.

## 4. Final implementation regression

The final implementation adds the creation check, assigns the
`bm_image_attach()` result, and uses one post-create cleanup point.

| Test | Result |
|---|---|
| One decoder, 25 calls/s, 30 minutes | 44,169 successful frames; 0.40 allocated B/frame; +0.0006 MiB/min |
| Four streams, 90 minutes | 538,001/538,005 successful calls; +0.00567 MiB/min combined allocated slope |
| Twelve processes, 22 hours | 23,812,506 successful frames; no OOM; no unexpected process loss |
| Final six hours of soak | Worst allocated slope +0.0130 MiB/min; worst RSS slope +0.0026 MiB/min |

## 5. Allocator high-water behavior

After the leak was removed, some processes had a one-time heap-count increase
and RSS step, followed by a long plateau.  Processes without a heap-count step
grew only 1.8-2.1 MiB RSS over 22 hours.  Significant RSS changes coincided
with the allocator step rather than with the decoded-frame count.

This bounded behavior is not evidence that the leak remains.  The original
defect is identified by `arena allocated` increasing linearly with every
successful frame; that signal disappears in the fixed arms.
