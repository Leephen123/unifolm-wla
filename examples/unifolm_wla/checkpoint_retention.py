"""Keep task/metric top-five union and the latest recoverable checkpoint."""
import json
import shutil
from pathlib import Path

METRICS = ('root_delta_rmse_m', 'body_xyz_rmse_cm', 'hand_rmse',
           'rotation_mean_deg', 'rotation_rmse_deg')
TASKS = ('full_pipeline',)


def select_top_five(results):
    rankings = {task: {metric: sorted(results, key=lambda s: (results[s][task][metric], s))[:5]
                      for metric in METRICS} for task in TASKS}
    keep = {step for task in rankings.values() for steps in task.values() for step in steps}
    return keep, rankings


def prune_checkpoints(out):
    out = Path(out).resolve()
    checkpoints = {int(p.name[5:]): p for p in (out / 'checkpoints').glob('step_*')
                   if (p / 'COMPLETE').is_file()}
    if not checkpoints:
        return
    # Rank ALL historical evaluations, including weights already pruned.
    results = {}
    for ready in (out / 'validation-optimized').glob('*/READY'):
        step = int(ready.parent.name)
        results[step] = {task: json.loads((out / 'validation-optimized' / str(step) / 'checkpoint.json').read_text())['metrics']
                         for task in TASKS}
    keep, rankings = select_top_five(results)
    keep.add(max(checkpoints))
    # Pending evaluations retain their checkpoint until evaluation completes.
    keep.update(set(checkpoints) - set(results))
    removed = []
    for step, path in checkpoints.items():
        if step not in keep:
            assert path.resolve().parent == out / 'checkpoints'
            shutil.rmtree(path)
            removed.append(step)
    record = {'rankings': rankings, 'retained': sorted(keep), 'removed_now': removed,
              'latest_recovery': max(checkpoints)}
    tmp = out / 'checkpoint-retention.json.tmp'
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out / 'checkpoint-retention.json')
