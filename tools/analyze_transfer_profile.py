"""Summarize instrumented timelines; CPU and GPU clocks stay separate.

Input: local/run-ID.json from watch_runs.py. First two calls are warmups.
DMA intervals include CUDA scheduling/event overhead, not only wire time.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from statistics import median


def analyze(payload, warmups=2):
    run = payload.get('run', payload)
    records = defaultdict(list)
    for line in run.get('build_log', '').splitlines():
        match = re.search(r'(GPU_SEGMENT|CPU_SEGMENT|DMA_PROBE) (.*)', line)
        if not match:
            continue
        row = dict(re.findall(r'(\w+)=([^\s]+)', match[2]))
        for key, value in row.items():
            if key != 'kind':
                row[key] = float(value) if '.' in value else int(value)
        if row['run'] > warmups:
            records[match[1]].append(row)
    metrics = defaultdict(list)
    gpu = defaultdict(list)
    for row in records['GPU_SEGMENT']:
        gpu[row['run']].append(row)
    # FHERMA may retain only the last 12,000 characters of build_log. Reject
    # partial calls rather than silently treating missing stages as zero.
    def complete(rows):
        parts = defaultdict(set)
        for row in rows:
            parts[row['kind']].add(row['part'])
        return (parts['prepare'] == set(range(4)) and parts['d2h'] == set(range(8))
                and all(parts[k] == {0} for k in ('forward', 'product', 'inverse', 'crt'))
                and (not parts['h2d'] or parts['h2d'] == set(range(4))))
    gpu = {call: rows for call, rows in gpu.items() if complete(rows)}
    for rows in gpu.values():
        stages = defaultdict(list)
        for row in rows:
            stages[row['kind']].append(row)
        for kind, parts in stages.items():
            duration = sum(p['end_us'] - p['start_us'] for p in parts)
            span = max(p['end_us'] for p in parts) - min(p['start_us'] for p in parts)
            metrics[f'gpu_{kind}_sum_us'].append(duration)
            metrics[f'gpu_{kind}_gaps_us'].append(span - duration)
        if stages['d2h'] and stages['crt']:
            metrics['gpu_crt_to_output_end_us'].append(max(p['end_us'] for p in stages['d2h']) - stages['crt'][-1]['end_us'])
            metrics['gpu_crt_to_first_dma_us'].append(min(p['start_us'] for p in stages['d2h']) - stages['crt'][-1]['end_us'])
        if stages['h2d']:
            overlap = sum(max(0, min(a['end_us'], b['end_us']) - max(a['start_us'], b['start_us']))
                          for a in stages['h2d'] for b in stages['prepare'])
            metrics['gpu_upload_prepare_overlap_us'].append(overlap)
        input_rows = stages['prepare'] + stages['h2d']
        metrics['gpu_input_span_us'].append(max(p['end_us'] for p in input_rows) - min(p['start_us'] for p in input_rows))
        metrics['gpu_ntt_crt_sum_us'].append(sum(p['end_us'] - p['start_us']
            for kind in ('forward', 'product', 'inverse', 'crt') for p in stages[kind]))
        metrics['gpu_full_span_us'].append(max(p['end_us'] for p in rows) - min(p['start_us'] for p in rows))
    cpu = defaultdict(list)
    for row in records['CPU_SEGMENT']:
        if row['run'] in gpu:
            cpu[(row['run'], row['kind'])].append(row)
    for (_, kind), rows in cpu.items():
        rows.sort(key=lambda p: p['part'])
        if kind == 'input':
            enqueue = sum(p['enqueued_us'] - p['ready_us'] for p in rows)
            metrics['cpu_input_enqueue_us'].append(enqueue)
            metrics['cpu_input_non_enqueue_us'].append(rows[-1]['enqueued_us'] - enqueue)
        else:
            metrics['cpu_output_wait_us'].append(sum(p['ready_us'] - p['wait_us'] for p in rows))
            metrics['cpu_output_copy_us'].append(sum(p['copied_us'] - p['ready_us'] for p in rows))
            metrics['cpu_result_complete_us'].append(rows[-1]['copied_us'])
    probes = defaultdict(lambda: defaultdict(list))
    for row in records['DMA_PROBE']:
        if row['run'] in gpu:
            key = tuple(row[k] for k in ('kind', 'parts', 'cached', 'flags', 'consume'))
            probes[key][row['run']].append(row)
    summaries = []
    for key, cases in sorted(probes.items()):
        gpu_us = median(median(p['gpu_us'] for p in case) for case in cases.values())
        host_us = median(median(p['host_us'] for p in case) for case in cases.values())
        size = next(iter(cases.values()))[0]['bytes']
        summaries.append(dict(zip(('kind', 'parts', 'cached', 'flags', 'consume'), key)) |
                         {'bytes': size, 'gpu_us': gpu_us, 'host_us': host_us,
                          'effective_GB_s': size / gpu_us / 1000})
    return {'run_id': run.get('id'), 'measured_calls': len(gpu),
            'included_call_ids': sorted(gpu), 'log_characters': len(run.get('build_log', '')),
            'median_us': {k: round(median(v), 3) for k, v in sorted(metrics.items())},
            'dma_probes': summaries, 'records': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.report.read_text()), args.warmups)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))
