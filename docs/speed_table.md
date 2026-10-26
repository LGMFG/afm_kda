GPU: NVIDIA A100-SXM-64GB  ·  versions: {'torch': '2.12.0+cu126', 'transformers': '5.8.1', 'flash-linear-attention': '0.5.2', 'flash_attn': '2.8.3'}

*Final (2026-10-26): batch 1/8 from lprod job 61585206; batch 32/64 from exclusive debug nodes (62299405/06/10/11, OOM points re-measured with `speed_bench.py` v1.2 in 62334075 + chain). Every OOM below is real (cache alone > 64 GB).*

## Prefill, batch 1 — ms per forward pass (speed-up over the teacher)

| model | params | 2k | 8k | 32k | 64k | 128k | log-log slope 16k–64k | crossover T |
|---|---|---|---|---|---|---|---|---|
| Teacher (AFM-4.5B) | 4.62 B | 98 | 396 | 2057 | 5756 | 18366 | 1.38 | — |
| KDA hybrid (R1) | 5.00 B | 111 (0.88×) | 417 (0.95×) | 1710 (1.20×) | 3828 (1.50×) | 9371 (1.96×) | 1.11 | 12k |
| GDN hybrid (R2) | 5.14 B | 109 (0.89×) | 411 (0.96×) | 1686 (1.22×) | 3777 (1.52×) | 9275 (1.98×) | 1.11 | 10k |
| KDA all-linear (R3) | 5.05 B | 115 (0.84×) | 424 (0.93×) | 1595 (1.29×) | 3184 (1.81×) | 6342 (2.90×) | 0.99 | 12k |
| GDN all-linear (R4) | 5.24 B | 113 (0.86×) | 416 (0.95×) | 1565 (1.31×) | 3127 (1.84×) | 6228 (2.95×) | 0.99 | 10k |

## Decode, batch 1 — ms per generated token (cache MiB)

| model | 2k | 8k | 32k | 64k | 128k |
|---|---|---|---|---|---|
| Teacher (AFM-4.5B) | 31.0 (148) | 31.0 (576) | 31.0 (2304) | 34.2 (4608) | 41.5 (9216) |
| KDA hybrid (R1) | 46.8 (79) | 46.8 (187) | 46.4 (619) | 47.6 (1195) | 49.0 (2347) |
| GDN hybrid (R2) | 44.7 (79) | 44.6 (187) | 44.6 (619) | 45.3 (1195) | 47.1 (2347) |
| KDA all-linear (R3) | 51.7 (57) | 51.5 (57) | 51.9 (57) | 51.8 (57) | 51.3 (57) |
| GDN all-linear (R4) | 49.5 (57) | 49.3 (57) | 49.3 (57) | 49.4 (57) | 49.5 (57) |

## Decode, batch 8 — ms per generated token (cache MiB)

| model | 2k | 8k | 32k | 64k | 128k |
|---|---|---|---|---|---|
| Teacher (AFM-4.5B) | 29.6 (1156) | 29.5 (4612) | 56.3 (18436) | 97.6 (36868) | OOM |
| KDA hybrid (R1) | 48.0 (575) | 48.1 (1439) | 48.2 (4895) | 48.2 (9503) | 56.1 (18719) |
| GDN hybrid (R2) | 45.8 (575) | 45.5 (1439) | 45.8 (4895) | 45.7 (9503) | 56.0 (18719) |
| KDA all-linear (R3) | 54.4 (381) | 54.2 (381) | 54.1 (381) | 54.1 (381) | 54.2 (381) |
| GDN all-linear (R4) | 51.5 (381) | 51.4 (381) | 51.6 (381) | 51.2 (381) | 51.2 (381) |

## Decode, batch 32 — ms per generated token (cache MiB)

| model | 2k | 8k | 32k | 64k | 128k |
|---|---|---|---|---|---|
| Teacher (AFM-4.5B) | 30.2 (4624) | 56.9 (18448) | OOM | OOM | OOM |
| KDA hybrid (R1) | 47.6 (2298) | 47.4 (5754) | 57.4 (19579) | 98.9 (38019) | OOM |
| GDN hybrid (R2) | 47.4 (2299) | 46.4 (5754) | 57.2 (19579) | 98.6 (38019) | OOM |
| KDA all-linear (R3) | 53.9 (1523) | 54.1 (1523) | 53.6 (1523) | 53.5 (1523) | 53.8 (1523) |
| GDN all-linear (R4) | 52.1 (1524) | 51.8 (1523) | 51.5 (1523) | 51.3 (1523) | 52.0 (1523) |

## Decode, batch 64 — ms per generated token (cache MiB)

| model | 2k | 8k | 32k | 64k | 128k |
|---|---|---|---|---|---|
| Teacher (AFM-4.5B) | 39.3 (9247) | 110.7 (36895) | OOM | OOM | OOM |
| KDA hybrid (R1) | 47.0 (4618) | 47.2 (11529) | 113.7 (39178) | OOM | OOM |
| GDN hybrid (R2) | 45.9 (4618) | 45.7 (11529) | 113.5 (39178) | OOM | OOM |
| KDA all-linear (R3) | 53.8 (3073) | 53.5 (3073) | 53.6 (3073) | 53.5 (3073) | 53.6 (3073) |
| GDN all-linear (R4) | 51.6 (3073) | 51.6 (3073) | 51.8 (3073) | 51.7 (3073) | 51.3 (3073) |

## Checks

| model | chunked = one-shot prefill: max logit diff (argmax) | next step: max diff (argmax) | logit scale | drift: T=8192 re-measured |
|---|---|---|---|---|
| Teacher (AFM-4.5B) | 0.125 (same) | 0.156 (same) | 9.1 | +0.1 % |
| KDA hybrid (R1) | 0.156 (same) | 0.188 (same) | 9.3 | -0.1 % |
| GDN hybrid (R2) | 0.156 (same) | 0.141 (same) | 9.4 | +0.1 % |
| KDA all-linear (R3) | 0.156 (same) | 0.125 (same) | 9.6 | +0.1 % |
| GDN all-linear (R4) | 0.234 (same) | 0.156 (same) | 11.7 | +0.1 % |