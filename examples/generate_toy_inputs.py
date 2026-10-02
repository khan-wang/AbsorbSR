"""Generate synthetic calibration inputs, unrelated to any benchmark run."""

import csv
from pathlib import Path

root = Path(__file__).resolve().parent
sites = [(step, block) for step in range(6) for block in range(4)]
risks = {site: 0.1 + site[0] * 0.01 + site[1] * 0.03 for site in sites}

with (root / "toy_sites.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["step", "layer", "endpoint_l1"])
    writer.writerows((step, block, round(risks[(step, block)], 6)) for step, block in sites)

with (root / "toy_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        "left_step", "left_layer", "right_step", "right_layer",
        "observed_endpoint_l1",
    ])
    for left, right in list(zip(sites, sites[5:] + sites[:5]))[:18]:
        cancellation = 0.2 + 0.03 * (left[0] == right[0])
        response = (risks[left] + risks[right]) * (1 - cancellation)
        writer.writerow([*left, *right, round(response, 6)])
