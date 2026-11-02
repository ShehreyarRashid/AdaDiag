# data/

The three datasets TVDiag releases, copied **unchanged** from
[WHU-AISE/TVDiag](https://github.com/WHU-AISE/TVDiag) at commit
[`cf6f334`](https://github.com/WHU-AISE/TVDiag/tree/cf6f334887036f363807bac1b8995c3b6f47c5db/data)
(28 April 2025). They are redistributed under TVDiag's MIT licence, reproduced in
[`LICENSE-TVDiag`](LICENSE-TVDiag). The data and its preparation are the TVDiag authors'
work; cite their paper (README section 10) if you use it.

| folder | system | incidents | instances | failure types | train / test | size |
|---|---|---|---|---|---|---|
| `gaia/` | MicroSS (GAIA) | 1099 | 10 | 5 | 160 / 939 | 23 MB |
| `hotel/` | Hotel Reservation | 200 | 18 | 5 | 122 / 78 | 6.5 MB |
| `sockshop/` | Sock Shop | 489 | 21 | 7 | 248 / 241 | 19 MB |

AIOps-22 is not included because TVDiag does not distribute it (see the top-level README,
section 4).

## Layout (identical for each dataset)

```
<dataset>/
├── label.csv            ground truth: incident id, root-cause instance, failure type, train/test split
├── raw/
│   ├── nodes.json       instance names per incident (one fixed ordering per dataset)
│   ├── edges.json       call-graph edges per incident
│   ├── metrics.json     metric alerts per incident   (3-sigma detection)
│   ├── traces.json      trace alerts per incident    (isolation forest, 30 s windows)
│   └── logs.json        log alerts per incident      (Drain3 templates)
└── tmp/
    ├── metric.pkl       128-d event embeddings per incident and instance (fastText)
    ├── trace.pkl
    └── log.pkl
```

`src/adadiag/data.py` reads exactly these files and asserts the counts in the table above
on every load.

## Integrity

`SHA256SUMS` lists the hash of every file as copied. To check nothing has changed:

```bash
cd data && sha256sum -c SHA256SUMS
```

`tests/test_data_integrity.py` runs the same check as part of `pytest`.

The `tmp/*.pkl` files are Python pickles. Loading a pickle can run code, so only load
files whose hashes match `SHA256SUMS`, i.e. the files TVDiag published.

## Description from the TVDiag repository

Quoted verbatim from TVDiag's README; "we" and "us" refer to the TVDiag authors.

> We conducted experiments on two datasets:
>
> - [GAIA](https://github.com/CloudWise-OpenSource/GAIA-DataSet). GAIA dataset records
>   metrics, traces, and logs of the MicroSS simulation system in July 2021, which consists
>   of ten microservices and some middleware such as Redis, MySQL, and Zookeeper. The
>   extracted events of GAIA can be accessible on [DiagFusion](https://arxiv.org/abs/2302.10512).
> - [AIOps-22](https://competition.aiops-challenge.com). The AIOps-22 dataset is derived from
>   the training data released by the AIOps 2022 Challenge, where failures at three levels
>   (node, service, and instance) were injected into a Web-based e-commerce platform
>   [Online-boutique](https://github.com/GoogleCloudPlatform/microservices-demo).
> - Sockshop and HotelReservation. We deployed two well-known microservice-based systems:
>   [Sockshop](https://github.com/microservices-demo/microservices-demo) and
>   [HotelReservation](https://github.com/delimitrou/DeathStarBench/tree/master/hotelReservation)
>   in a Kubernetes cluster. We induced diverse application failures using Chaos Mesh and
>   gathered system telemetry data for testing. Due to storage constraints, we only provide
>   extracted alert datasets. You may contact us via email to request the raw data of two
>   systems.

DiagFusion, linked above as a preprint, is published as: S. Zhang et al., "Robust Failure
Diagnosis of Microservice System Through Multimodal Data," *IEEE Transactions on Services
Computing*, 16(6), 3851–3864, 2023,
[doi:10.1109/TSC.2023.3290018](https://doi.org/10.1109/TSC.2023.3290018).
