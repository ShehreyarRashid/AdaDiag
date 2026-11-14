"""Numbers published in the TVDiag paper, for the comparison in results/paper_comparison/.

Source: S. Xie, J. Wang, H. He, Z. Wang, Y. Zhao, N. Zhang, B. Li. "TVDiag: A Task-oriented
and View-invariant Failure Diagnosis Framework for Microservice-based Systems with
Multimodal Data." ACM TOSEM 35(2), Article 40, January 2026. doi:10.1145/3734868.

Two kinds of entry, never mixed:

* TABLE values are transcribed exactly as printed (Tables 3-7; page numbers below).
* FIGURE values are READ OFF the published figures (Figs. 15-18), which print no numbers.
  They were measured from the page rendered at 250 dpi: axes calibrated on the frame and
  gridlines, bar tops / line markers located by colour. Checked against the bars the paper
  also prints in a table (Fig. 17 "w/ Trace", "w/ Log", "TVDiag" = Table 7 / Table 6):
  largest error 0.003. Treat them as +/- 0.005.

The paper's datasets are A = GAIA, B = AIOps-22, C = Sock Shop, D = Hotel Reservation
(Sec. 5.1). B is not distributed with TVDiag's code, so it has no counterpart here.
"""
from __future__ import annotations

DATASET_KEY = {"A": "gaia", "B": "aiops22", "C": "sockshop", "D": "hotel"}
DATASET_NAME = {"A": "GAIA", "B": "AIOps-22", "C": "Sock Shop", "D": "Hotel Reservation"}

# --------------------------------------------------------------------------- Table 3, p. 40:19
TABLE3 = {
    "A": dict(microservices=5, instances=10, failure_types=5,
              records=dict(metric=217_461_390, trace=3_084_066, log=87_974_577),
              types=dict(metric=3_320, trace=122, log=49)),
    "B": dict(microservices=10, instances=40, failure_types=9,
              records=dict(metric=25_914_595, trace=63_283_650, log=95_003_038),
              types=dict(metric=404, trace=63, log=119)),
    "C": dict(microservices=14, instances=21, failure_types=7,
              records=dict(metric=77_376_704, trace=7_297_688, log=20_365_334),
              types=dict(metric=293, trace=143, log=422)),
    "D": dict(microservices=18, instances=18, failure_types=5,
              records=dict(metric=47_809_236, trace=1_218_830, log=258_927),
              types=dict(metric=393, trace=12, log=112)),
}

# --------------------------------------------------------------------------- Table 4, p. 40:22
#: dataset -> [(modality, method, p-value vs TVDiag, HR@1, HR@3, Avg@3, MRR@3)]
TABLE4_COLS = ("HR@1", "HR@3", "Avg@3", "MRR@3")
TABLE4 = {
    "A": [("Metric", "MicroRCA", "3.4e-05", 0.181, 0.385, 0.287, 0.268),
          ("Metric", "MicroHECL", "2.4e-06", 0.118, 0.233, 0.177, 0.167),
          ("Trace", "TraceRCA", "1.3e-05", 0.192, 0.364, 0.291, 0.270),
          ("Trace", "MicroRank", "8.6e-05", 0.192, 0.419, 0.308, 0.288),
          ("Multimodal", "Eadro", "1.6e-03", 0.304, 0.592, 0.473, 0.437),
          ("Multimodal", "DiagFusion", "3.1e-02", 0.404, 0.801, 0.636, 0.586),
          ("Multimodal", "RF-RCL", "3.4e-02", 0.573, 0.815, 0.719, 0.686),
          ("Multimodal", "TVDiag", "-", 0.759, 0.909, 0.844, 0.827)],
    "B": [("Metric", "MicroRCA", "6.7e-07", 0.058, 0.155, 0.107, 0.099),
          ("Metric", "MicroHECL", "6.8e-07", 0.044, 0.142, 0.094, 0.085),
          ("Trace", "TraceRCA", "2.8e-09", 0.150, 0.204, 0.183, 0.176),
          ("Trace", "MicroRank", "4.4e-09", 0.171, 0.227, 0.205, 0.197),
          ("Multimodal", "Eadro", "6.3e-04", 0.226, 0.478, 0.367, 0.339),
          ("Multimodal", "DiagFusion", "4.3e-08", 0.205, 0.273, 0.239, 0.229),
          ("Multimodal", "RF-RCL", "7.5e-03", 0.635, 0.829, 0.739, 0.719),
          ("Multimodal", "TVDiag", "-", 0.894, 0.941, 0.924, 0.917)],
    "C": [("Metric", "MicroRCA", "6.4E-06", 0.063, 0.231, 0.145, 0.132),
          ("Metric", "MicroHECL", "1.8E-05", 0.054, 0.245, 0.147, 0.132),
          ("Trace", "TraceRCA", "1.3E-07", 0.116, 0.220, 0.171, 0.161),
          ("Trace", "MicroRank", "1.7E-06", 0.054, 0.200, 0.140, 0.121),
          ("Multimodal", "Eadro", "1.4E-04", 0.191, 0.415, 0.317, 0.291),
          ("Multimodal", "DiagFusion", "9.0E-04", 0.519, 0.710, 0.612, 0.597),
          ("Multimodal", "RF-RCL", "5.1E-03", 0.622, 0.817, 0.715, 0.701),
          ("Multimodal", "TVDiag", "-", 0.863, 0.954, 0.913, 0.903)],
    "D": [("Metric", "MicroRCA", "7.9E-04", 0.150, 0.460, 0.317, 0.285),
          ("Metric", "MicroHECL", "2.3E-04", 0.129, 0.387, 0.269, 0.242),
          ("Trace", "TraceRCA", "2.9E-07", 0.151, 0.258, 0.197, 0.192),
          ("Trace", "MicroRank", "1.2E-04", 0.172, 0.398, 0.283, 0.265),
          ("Multimodal", "Eadro", "2.6E-03", 0.244, 0.590, 0.419, 0.389),
          ("Multimodal", "DiagFusion", "4.5E-03", 0.513, 0.744, 0.645, 0.618),
          ("Multimodal", "RF-RCL", "4.6E-02", 0.744, 0.872, 0.804, 0.795),
          ("Multimodal", "TVDiag", "-", 0.821, 0.923, 0.876, 0.866)],
}

# --------------------------------------------------------------------------- Table 5, p. 40:23
#: dataset -> [(modality, method, p-value, Precision, Recall, F1)]
TABLE5 = {
    "A": [("Log", "LogCluster", "1.4E-02", 0.888, 0.915, 0.901),
          ("Multimodal", "DiagFusion", "3.3E-04", 0.852, 0.838, 0.84),
          ("Multimodal", "Medicine", "4.1E-06", 0.669, 0.667, 0.648),
          ("Multimodal", "DT-FTI", "9.4E-03", 0.889, 0.892, 0.89),
          ("Multimodal", "SVM-FTI", "9.9E-02", 0.883, 0.935, 0.908),
          ("Multimodal", "LGB-FTI", "5.2E-02", 0.892, 0.928, 0.907),
          ("Multimodal", "TVDiag", "-", 0.925, 0.948, 0.936)],
    "B": [("Log", "LogCluster", "1.1E-02", 0.434, 0.159, 0.147),
          ("Multimodal", "DiagFusion", "3.4E-04", 0.533, 0.489, 0.505),
          ("Multimodal", "Medicine", "4.7E-04", 0.307, 0.244, 0.244),
          ("Multimodal", "DT-FTI", "5.0E-05", 0.448, 0.42, 0.422),
          ("Multimodal", "SVM-FTI", "3.6E-03", 0.807, 0.784, 0.774),
          ("Multimodal", "LGB-FTI", "7.0E-04", 0.604, 0.58, 0.556),
          ("Multimodal", "TVDiag", "-", 0.882, 0.871, 0.876)],
    "C": [("Log", "LogCluster", "1.9E-06", 0.233, 0.228, 0.213),
          ("Multimodal", "DiagFusion", "7.9E-05", 0.338, 0.332, 0.328),
          ("Multimodal", "Medicine", "5.0E-04", 0.497, 0.469, 0.454),
          ("Multimodal", "DT-FTI", "6.4E-05", 0.376, 0.365, 0.366),
          ("Multimodal", "SVM-FTI", "3.2E-03", 0.384, 0.315, 0.282),
          ("Multimodal", "LGB-FTI", "9.7E-04", 0.442, 0.39, 0.388),
          ("Multimodal", "TVDiag", "-", 0.649, 0.622, 0.636)],
    "D": [("Log", "LogCluster", "4.5E-03", 0.396, 0.282, 0.212),
          ("Multimodal", "DiagFusion", "3.2E-05", 0.618, 0.603, 0.600),
          ("Multimodal", "Medicine", "1.9E-02", 0.782, 0.725, 0.728),
          ("Multimodal", "DT-FTI", "5.2E-03", 0.676, 0.603, 0.598),
          ("Multimodal", "SVM-FTI", "9.5E-03", 0.488, 0.628, 0.540),
          ("Multimodal", "LGB-FTI", "4.1E-02", 0.809, 0.769, 0.764),
          ("Multimodal", "TVDiag", "-", 0.836, 0.808, 0.822)],
}

# --------------------------------------------------------------------------- Table 6, p. 40:25
#: variant -> {dataset: (HR@1, HR@3, Avg@3, F1)}
TABLE6 = {
    "w/o AUG": {"A": (0.675, 0.879, 0.794, 0.905), "B": (0.882, 0.959, 0.922, 0.838)},
    "w/o TO":  {"A": (0.754, 0.911, 0.843, 0.920), "B": (0.835, 0.953, 0.902, 0.864)},
    "w/o CM":  {"A": (0.757, 0.908, 0.845, 0.931), "B": (0.888, 0.959, 0.929, 0.868)},
    "w/o DW":  {"A": (0.757, 0.908, 0.843, 0.935), "B": (0.847, 0.941, 0.900, 0.858)},
    "TVDiag":  {"A": (0.759, 0.909, 0.844, 0.936), "B": (0.894, 0.941, 0.924, 0.876)},
}

# --------------------------------------------------------------------------- Table 7, p. 40:26
#: dataset -> [(extractor, modality, HR@1, HR@3, Avg@3, F1)]. IForest (trace) and rule-based
#: (log) are TVDiag's own extractors; 3-sigma and sampling are DiagFusion's.
TABLE7 = {
    "A": [("IForest", "Trace", 0.410, 0.583, 0.506, 0.732),
          ("3-sigma", "Trace", 0.398, 0.604, 0.504, 0.731),
          ("Rule-based", "Log", 0.337, 0.639, 0.518, 0.945),
          ("Sampling", "Log", 0.301, 0.644, 0.508, 0.877)],
    "B": [("IForest", "Trace", 0.435, 0.647, 0.537, 0.528),
          ("3-sigma", "Trace", 0.102, 0.205, 0.155, 0.221),
          ("Rule-based", "Log", 0.235, 0.300, 0.274, 0.567),
          ("Sampling", "Log", 0.080, 0.182, 0.125, 0.194)],
}

# --------------------------------------------------------------------------- figures (read off)
METRICS4 = ("HR@1", "HR@3", "Avg@3", "F1")

#: Fig. 15, p. 40:27 -- alert-field ablation, dataset A, single modality. (HR@1, HR@3, Avg@3, F1)
FIG15_A = {
    "w/o opName":      (0.403, 0.586, 0.504, 0.694),
    "w/o abType":      (0.376, 0.582, 0.483, 0.612),
    "w/ Trace":        (0.410, 0.583, 0.506, 0.732),     # = Table 7, IForest (exact)
    "w/o abDirection": (0.444, 0.783, 0.621, 0.596),
    "w/ Metric":       (0.465, 0.776, 0.633, 0.596),
}

#: Fig. 17, p. 40:30 -- discarding modalities, dataset A. (HR@1, HR@3, Avg@3, F1)
#: w/ Trace and w/ Log are exact (they are Table 7's IForest and rule-based rows); TVDiag is
#: Table 6. w/o Metric / w/o Trace HR@1 agree with the text's "27.4% ~ 29.4%" absolute drop
#: (0.759 - 0.485 = 0.274, 0.759 - 0.465 = 0.294), and w/o Log F1 with its "21.6%".
FIG17_A = {
    "w/ Metric":  (0.465, 0.776, 0.632, 0.595),
    "w/ Trace":   (0.410, 0.583, 0.506, 0.732),
    "w/ Log":     (0.337, 0.639, 0.518, 0.945),
    "w/o Metric": (0.485, 0.646, 0.571, 0.944),
    "w/o Trace":  (0.465, 0.827, 0.663, 0.926),
    "w/o Log":    (0.694, 0.862, 0.793, 0.720),
    "TVDiag":     (0.759, 0.909, 0.844, 0.936),
}

#: Fig. 16, p. 40:28 -- hyper-parameter sensitivity, dataset A.
FIG16_A = {
    "p": {"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
          "HR@1": [0.756, 0.761, 0.764, 0.765, 0.754, 0.753, 0.754, 0.707, 0.711],
          "HR@3": [0.904, 0.909, 0.902, 0.908, 0.907, 0.900, 0.897, 0.881, 0.892],
          "Avg@3": [0.836, 0.844, 0.844, 0.843, 0.843, 0.837, 0.834, 0.808, 0.814],
          "F1": [0.930, 0.936, 0.940, 0.940, 0.941, 0.935, 0.937, 0.930, 0.944]},
    "omega": {"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              "HR@1": [0.761, 0.754, 0.759, 0.750, 0.745, 0.746, 0.751, 0.754, 0.756],
              "HR@3": [0.910, 0.904, 0.899, 0.898, 0.906, 0.905, 0.902, 0.902, 0.899],
              "Avg@3": [0.843, 0.837, 0.840, 0.836, 0.839, 0.839, 0.840, 0.841, 0.838],
              "F1": [0.936, 0.936, 0.926, 0.925, 0.928, 0.928, 0.923, 0.926, 0.928]},
    "tau": {"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "HR@1": [0.756, 0.761, 0.761, 0.752, 0.755, 0.757, 0.754, 0.751, 0.751],
            "HR@3": [0.904, 0.913, 0.910, 0.908, 0.906, 0.905, 0.906, 0.905, 0.905],
            "Avg@3": [0.840, 0.846, 0.844, 0.840, 0.842, 0.842, 0.842, 0.840, 0.840],
            "F1": [0.935, 0.939, 0.937, 0.929, 0.928, 0.928, 0.926, 0.922, 0.922]},
    "l": {"x": [1, 2, 3, 4],
          "HR@1": [0.732, 0.761, 0.735, 0.713],
          "HR@3": [0.899, 0.910, 0.906, 0.890],
          "Avg@3": [0.826, 0.844, 0.833, 0.813],
          "F1": [0.917, 0.936, 0.940, 0.923]},
}
#: The paper's stated best values (Sec. 5.4 and the RQ3 answer, p. 40:27-29).
FIG16_BEST = {"p": 0.2, "omega": 0.1, "tau": 0.2, "l": 2}

#: Fig. 18, p. 40:31 -- quartiles of each modality's SHAP-value CDF, dataset A.
#: The medians sum to ~1 per task (FTI 1.00, RCL 0.96): the paper's values are shares.
FIG18_A = {
    "fti": {"metric": (0.294, 0.357, 0.430), "trace": (0.213, 0.259, 0.318),
            "log": (0.308, 0.388, 0.441)},
    "rcl": {"metric": (0.266, 0.330, 0.406), "trace": (0.346, 0.441, 0.556),
            "log": (0.147, 0.189, 0.283)},
}

#: Default hyper-parameters stated in Sec. 5.1 (p. 40:21).
DEFAULTS = {"temperature": 0.3, "graph_layers": 2, "lr": 1e-3, "weight_decay": 1e-4}
