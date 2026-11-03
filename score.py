"""Score a selected, complete set of LongBench or RULER prediction files."""
import argparse
import csv
import json
import re
from pathlib import Path
from benchmarks.tasks import LONGBENCH_TASKS, RULER_TOKENS


def score_results(benchmark, results_dir, method='echo', tasks=None):
    tasks = tasks or (LONGBENCH_TASKS if benchmark == 'longbench' else list(RULER_TOKENS))
    scores = {}
    for task in tasks:
        path = Path(results_dir) / task / f'{method}.json'
        with path.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            raise ValueError(f'Empty prediction file: {path}')
        predictions = [row['pred'] for row in rows]
        answers = [row['answers'] for row in rows]
        if benchmark == 'longbench':
            from eval import scorer
            score = scorer(task, predictions, answers, rows[0].get('all_classes', []))
        else:
            from benchmarks.ruler_metrics import string_match_all, string_match_part
            predictions = [re.sub(r'[\x00-\x1f]', '\n', (pred or '').strip()) for pred in predictions]
            metric = string_match_part if task.startswith('qa_') else string_match_all
            score = metric(predictions, answers)
        scores[task] = {'score': score, 'samples': len(rows)}
    result = {'benchmark': benchmark, 'method': method, 'tasks': scores,
              'average': sum(row['score'] for row in scores.values()) / len(scores)}
    output = Path(results_dir) / f'scores_{method}.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['task', 'score', 'samples'])
        writer.writerows((name, row['score'], row['samples']) for name, row in scores.items())
        writer.writerow(['average', result['average'], ''])
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=['longbench','ruler'], required=True)
    p.add_argument('--results-dir', type=Path, required=True)
    p.add_argument('--method', default='echo')
    p.add_argument('--tasks', nargs='+')
    a = p.parse_args()
    allowed = LONGBENCH_TASKS if a.benchmark == 'longbench' else list(RULER_TOKENS)
    if a.tasks and any(task not in allowed for task in a.tasks):
        p.error('Unknown benchmark task')
    print(json.dumps(score_results(a.benchmark, a.results_dir, a.method, a.tasks), indent=2))
