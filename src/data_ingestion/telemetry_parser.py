"""
Telemetry and Medical Report Parser for AthleteIQ.
Converts tabular GPS/workload CSV records and clinical physio notes into rich semantic narratives.
"""

import os
import glob
from typing import List, Dict, Any, Optional
import pandas as pd


class TelemetryParser:
    """
    Parses private athlete telemetry (GPS logs, bowling workloads, ACWR metrics)
    and confidential physiotherapist reports into natural language narratives.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        parsed_output_path: Optional[str] = None,
        medical_reports_dir: Optional[str] = None,
    ):
        self.csv_path = csv_path or os.path.join(
            "data", "private_telemetry", "raw_csvs", "player_daily_metrics.csv"
        )
        self.parsed_output_path = parsed_output_path or os.path.join(
            "data", "private_telemetry", "parsed_txt", "player_daily_summaries.txt"
        )
        self.medical_reports_dir = medical_reports_dir or os.path.join(
            "data", "private_telemetry", "medical_reports"
        )

    @staticmethod
    def get_acwr_interpretation(acwr: float) -> str:
        """Determines clinical injury risk zone based on Tim Gabbett's ACWR model."""
        if acwr < 0.80:
            return "Under-prepared / sub-optimal training baseline (< 0.80)"
        elif 0.80 <= acwr <= 1.30:
            return "Optimal high-performance sweet spot (0.80 - 1.30)"
        elif 1.30 < acwr < 1.50:
            return "Elevated risk monitoring zone (1.30 - 1.49)"
        else:
            return "CRITICAL DANGER ZONE SPIKE (>= 1.50) with 2-4x elevated lumbar stress risk"

    def record_to_narrative(self, row: pd.Series) -> str:
        """Converts an individual telemetry dataframe row into a descriptive sports-science narrative."""
        date_str = str(row.get("date", "Unknown Date"))
        player_id = str(row.get("player_id", "Unknown Player"))
        player_name = str(row.get("player_name", "Athlete"))
        role = str(row.get("role", "Pace Bowler"))
        session = str(row.get("session_type", "Training"))
        overs = float(row.get("overs_bowled", 0))
        balls = int(row.get("balls_bowled", int(overs * 6)))
        dist = float(row.get("total_distance_km", 0))
        hsr = int(row.get("high_speed_running_m", 0))
        sprint = int(row.get("sprint_distance_m", 0))
        load = float(row.get("player_load", 0))
        vel = float(row.get("run_up_velocity_kmh", 0))
        acute = float(row.get("acute_workload", 0))
        chronic = float(row.get("chronic_workload", 0))
        acwr = float(row.get("acwr", 0))
        rpe = int(row.get("rpe", 0))
        sleep = float(row.get("sleep_hours", 0))
        lumbar = int(row.get("lumbar_soreness", 0))
        hamstring = int(row.get("hamstring_soreness", 0))

        acwr_status = self.get_acwr_interpretation(acwr)

        narrative = (
            f"TELEMETRY LOG: On {date_str}, {player_name} ({player_id}, Role: {role}) "
            f"participated in a {session}. "
            f"Bowling Volume: {overs:.1f} overs bowled ({balls} deliveries) with an approach run-up velocity of {vel:.1f} km/h. "
            f"GPS Movement Metrics: Covered {dist:.2f} km total distance, {hsr} m of High-Speed Running (>19.8 km/h), "
            f"{sprint} m of sprinting (>25.2 km/h), and accumulated a total PlayerLoad of {load:.1f} units. "
            f"Workload Assessment: 7-day acute bowling load was {acute:.1f} overs, compared against a 28-day chronic baseline of {chronic:.1f} overs. "
            f"Calculated Acute:Chronic Workload Ratio (ACWR): {acwr:.2f} [{acwr_status}]. "
            f"Wellness & Exertion: Session Rate of Perceived Exertion (RPE): {rpe}/10. Reported sleep: {sleep:.1f} hours. "
            f"Subjective soreness ratings: Lumbar spine soreness {lumbar}/10, Hamstring soreness {hamstring}/10."
        )
        return narrative

    def parse_csv_telemetry(self) -> List[Dict[str, Any]]:
        """Parses the raw CSV file into narrative documents and saves summaries text."""
        if not os.path.exists(self.csv_path):
            print(f"[TelemetryParser] CSV file not found: {self.csv_path}")
            return []

        df = pd.read_csv(self.csv_path)
        documents = []
        all_narratives = []

        for _, row in df.iterrows():
            narrative = self.record_to_narrative(row)
            all_narratives.append(narrative)

            doc_metadata = {
                "source": self.csv_path,
                "file_name": os.path.basename(self.csv_path),
                "document_type": "telemetry_summary",
                "date": str(row.get("date", "")),
                "player_id": str(row.get("player_id", "")),
                "player_name": str(row.get("player_name", "")),
                "role": str(row.get("role", "")),
                "acwr": float(row.get("acwr", 0.0)),
                "overs_bowled": float(row.get("overs_bowled", 0.0)),
                "lumbar_soreness": int(row.get("lumbar_soreness", 0)),
                "domain": "private_telemetry"
            }

            documents.append({
                "page_content": narrative,
                "metadata": doc_metadata
            })

        # Save concatenated summaries to parsed_output_path
        os.makedirs(os.path.dirname(self.parsed_output_path), exist_ok=True)
        with open(self.parsed_output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_narratives))
        print(f"[TelemetryParser] Saved parsed telemetry summaries to {self.parsed_output_path}")

        return documents

    def parse_medical_reports(self) -> List[Dict[str, Any]]:
        """Scans the medical_reports directory and loads confidential clinical physio notes."""
        if not os.path.exists(self.medical_reports_dir):
            return []

        docs = []
        report_files = glob.glob(os.path.join(self.medical_reports_dir, "*.txt"))
        for rep_path in report_files:
            try:
                with open(rep_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                file_name = os.path.basename(rep_path)
                player_id = "Player 9" if "player_9" in file_name.lower() else "Unknown Athlete"

                doc_metadata = {
                    "source": rep_path,
                    "file_name": file_name,
                    "document_type": "medical_report",
                    "player_id": player_id,
                    "domain": "confidential_clinical_physio"
                }

                docs.append({
                    "page_content": content,
                    "metadata": doc_metadata
                })
            except Exception as e:
                print(f"[TelemetryParser] Error reading {rep_path}: {e}")

        print(f"[TelemetryParser] Loaded {len(docs)} medical physio reports.")
        return docs

    def load_all_telemetry_and_reports(self) -> List[Dict[str, Any]]:
        """Loads both CSV telemetry logs and medical physio notes."""
        telemetry_docs = self.parse_csv_telemetry()
        medical_docs = self.parse_medical_reports()
        return telemetry_docs + medical_docs
