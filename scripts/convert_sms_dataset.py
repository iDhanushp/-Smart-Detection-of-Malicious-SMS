#!/usr/bin/env python3
"""convert_sms_dataset.py
Usage:
  python convert_sms_dataset.py exported.csv output_dir

Reads a CSV exported from the sms_extractor app (id,address,body,date[,label])
and produces two text files:
  legit.txt – one legitimate message per line
  spam.txt  – one spam message per line
If the 'label' column is missing, all messages are written to legit.txt and
spam.txt is created empty; you can then manually move spam lines and retrain.
"""
import csv
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print('Usage: python convert_sms_dataset.py input.csv output_dir')
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    legit_path = out_dir / 'legit.txt'
    spam_path = out_dir / 'spam.txt'

    legit_msgs = []
    spam_msgs = []

    with in_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'body' not in reader.fieldnames:
            print('Input CSV missing "body" column')
            sys.exit(1)
        has_label = 'label' in reader.fieldnames
        for row in reader:
            msg = row['body'].replace('\n', ' ').strip()
            if has_label:
                label = row['label'].lower().strip()
                if label in {'spam', 'fraud', 'scam'}:
                    spam_msgs.append(msg)
                else:
                    legit_msgs.append(msg)
            else:
                legit_msgs.append(msg)

    legit_path.write_text('\n'.join(legit_msgs), encoding='utf-8')
    spam_path.write_text('\n'.join(spam_msgs), encoding='utf-8')
    print(f'Wrote {len(legit_msgs)} legit lines to {legit_path}')
    print(f'Wrote {len(spam_msgs)} spam  lines to {spam_path}')

if __name__ == '__main__':
    main() 