import csv
from pathlib import Path

import duckdb

from learnpulse.reference_model import FEATURES


def make_table(root:Path,students=30):
 rows=[]
 for s in range(students):
  for day in (28,56):
   r={f:float((s+day)%9+1) for f in FEATURES};r["historical_windows_available"]=int(r["historical_windows_available"]);r["historical_average_active_days"]=float(s%7);r["current_inactivity_gap"]=None if s%7==0 else float(s%5);r.update(future_inactivity=bool(s%2),student_id=s,module="AAA" if s<students/2 else "BBB",presentation="2013J" if s%3 else "2014J",observation_day=day);rows.append(r)
 csvp=root/"input.csv";p=root/"model.parquet"
 with csvp.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 dest=p.resolve().as_posix().replace("'","''")
 with duckdb.connect() as c:c.execute(f"copy (select * from read_csv_auto(?,header=true)) to '{dest}' (format parquet)",[str(csvp)])
 return p,rows
