import time
import os
import csv
import glob
import pandas as pd
from datetime import datetime
from database import get_connection
from special_days import is_special_day

class HistoryAggregator:
    def __init__(self, input_pattern, output_file, refresh_interval=5):
        self.input_pattern    = input_pattern
        self.output_file      = output_file
        self.refresh_interval = refresh_interval

        # Track how many rows we already read per file
        # so next cycle we only read new ones
        self._last_row_count = {}

        # Create output file with header if it doesn't exist
        self._init_output()

    def _init_output(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        if not os.path.exists(self.output_file):
            with open(self.output_file, "w", newline="") as f:
                pass  # empty file, header written on first append

    def _get_history_files(self):
        files = glob.glob(self.input_pattern)
        return [f for f in files if "global" not in f]

    def _read_new_rows(self, filepath):
        """Read only rows added since last cycle."""
        last_count = self._last_row_count.get(filepath, 0)

        try:
            df = pd.read_csv(filepath)  # read full file
        except Exception as e:
            print(f"  ⚠️  Error reading {filepath}: {e}")
            return pd.DataFrame()

        total_rows = len(df)

        if total_rows <= last_count:
            return pd.DataFrame()  # no new rows since last cycle

        new_rows = df.iloc[last_count:]          # only truly new rows
        self._last_row_count[filepath] = total_rows  # update pointer
        return new_rows

    # Connection avec la base de données
    def _insert_to_db(self, df):
        if df.empty:
            return 0

        conn = get_connection()
        cur  = conn.cursor()

        try:
            # Compute one summary row for the entire 30-min batch
            ts         = pd.to_datetime(df["timestamp"].iloc[-1])  # last timestamp
            parking_id = df["parking_id"].iloc[0]
            special_day = is_special_day(ts)

            cur.execute("""
                INSERT INTO occupancy_history (
                    parking_id, timestamp,
                    free_spots, capacity,
                    occupancy_rate, hour,
                    is_weekend, free_spot_ids, is_special_day
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                parking_id,
                ts,
                round(df["free_spots"].mean()),        # average free spots
                int(df["capacity"].iloc[0]),            # capacity doesn't change
                round(float(df["occupancy_rate"].mean()), 3),  # average occupancy
                int(ts.hour),
                bool(df["is_weekend"].iloc[0]),
                df["free_spot_ids"].iloc[-1],           # latest snapshot
                special_day,
            ))

            conn.commit()
        except Exception as e:
            print(f" Insert error: {e}")
            return 0
        finally:
            cur.close()
            conn.close()

        return 1  # one row inserted per 30-min batch


    def aggregate(self):
        files    = self._get_history_files()
        total_new = 0

        for filepath in files:
            new_rows = self._read_new_rows(filepath)

            if new_rows.empty:
                continue

            inserted = self._insert_to_db(new_rows)
            total_new += inserted

        if total_new > 0:
            now = datetime.now().strftime("%H:%M:%S")
            total = sum(self._last_row_count.values())
            print(f" [{now}] +{total_new} new rows appended "
                  f"→ global total: {total} rows")

    def get_summary(self):
        if not os.path.exists(self.output_file):
            print(" No global file yet")
            return

        try:
            df      = pd.read_csv(self.output_file)
            counts  = df.groupby("parking_id").size()

            print(f"\n  {'─'*45}")
            print(f"  📊 Global History Summary")
            print(f"  {'─'*45}")
            for pid, n in counts.items():
                bar = "█" * min(n // 10, 20)
                print(f"  Parking {str(pid):<5} → {n:>5} rows  {bar}")
            print(f"  {'─'*45}")
            print(f"  Total : {len(df)} rows")
            print(f"  File  : {self.output_file}\n")
        except Exception as e:
            print(f" Could not read global file: {e}")

    def run(self):
        print(f" Aggregator started "
              f"(checking every {self.refresh_interval}s)\n")

        while True:
            self.aggregate()
            time.sleep(self.refresh_interval)