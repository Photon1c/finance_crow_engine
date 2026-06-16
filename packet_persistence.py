import csv
from datetime import datetime, timezone
from pathlib import Path

LEDGER_FIELDS = [
    "timestamp",
    "trade_id",
    "ticker",
    "instrument",
    "status",
    "failure_type",
    "failure_score",
    "weighted_score",
    "theta_risk",
    "cvd_confirmation",
    "direction_score",
    "timing_score",
    "magnitude_score",
    "volatility_score",
    "catalyst_score",
    "exit_score",
    "absorption_capacity_mod",
    "selling_pressure_mod",
    "model_update_note",
]


def outputs_dir(base_dir) -> Path:
    path = Path(base_dir) / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ledger_path(base_dir, ticker: str) -> Path:
    return outputs_dir(base_dir) / f"recursive_packets_{ticker.upper()}.csv"


def report_path(base_dir, ticker: str) -> Path:
    return outputs_dir(base_dir) / f"latest_report_{ticker.upper()}.md"


def packet_to_row(packet: dict) -> dict:
    row = {field: packet.get(field, "") for field in LEDGER_FIELDS}
    row["cvd_confirmation"] = 1 if packet.get("cvd_confirmation") in (True, 1, "1", "True") else 0
    return row


def append_packet(base_dir, ticker: str, packet: dict) -> Path:
    path = ledger_path(base_dir, ticker)
    row = packet_to_row(packet)
    if "timestamp" not in packet or not packet["timestamp"]:
        row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return path


def save_report(base_dir, ticker: str, markdown_text: str) -> Path:
    path = report_path(base_dir, ticker)
    path.write_text(markdown_text, encoding="utf-8")
    return path
