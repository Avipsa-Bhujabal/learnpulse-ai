import argparse
import json
from pathlib import Path

from learnpulse.batch_scoring import batch_score


def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--artifact-dir",type=Path,required=True);p.add_argument("--batch-size",type=int,default=4096);p.add_argument("--replace",action="store_true");a=p.parse_args(argv)
 print(json.dumps(batch_score(a.input,a.output,a.artifact_dir,a.batch_size,a.replace),indent=2))
if __name__=="__main__":main()
