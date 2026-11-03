"""Aggregate repeated switching measurements while retaining their protocols."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

FIELDS = ('mode', 'prompt_length', 'batch_size', 'switch_after', 'output_length',
          'cache_representation', 'native_fullkv_prefix', 'warmup', 'empty_cache_each_layer',
          'profile_components', 'model_path', 'mlp_path', 'block_size', 'total_save_dim',
          'sink_size', 'local_window_size', 'seed')


def summarize(root):
    groups = defaultdict(list)
    for path in root.glob('run_*/*.json'):
        row = json.loads(path.read_text())
        groups[tuple(row.get(field) for field in FIELDS)].append(row)
    if not groups:
        raise ValueError(f'No switching results under {root}')
    result = []
    for key, rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        good = [row for row in rows if row['status'] == 'completed']
        def average(field, scale=1):
            values = [row[field] * scale for row in good if row.get(field) is not None]
            return mean(values) if values else None
        def cache_average(field):
            values = [row[field]['total_bytes'] / 1024**3 for row in good if row.get(field)]
            return mean(values) if values else None
        result.append(dict(zip(FIELDS, key),
                           completed=len(good), attempted=len(rows),
                           oom=sum(row['status']=='oom' for row in rows),
                           switch_ms=average('switch_latency_s',1000), tpot_ms=average('steady_tpot_s',1000),
                           pre_switch_cache_gib=cache_average('pre_switch_cache_bytes'),
                           post_switch_cache_gib=cache_average('post_switch_cache_bytes'),
                           final_cache_gib=cache_average('final_cache_bytes'),
                           predictor_gib=average('predictor_parameter_bytes',1/1024**3),
                           max_total_peak_gib=max((row['end_to_end_peak_gib'] for row in good
                                                  if row.get('end_to_end_peak_gib') is not None), default=None)))
    (root / 'switching_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root / 'switching_summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,required=True)
    print(json.dumps(summarize(p.parse_args().results_root),indent=2))
