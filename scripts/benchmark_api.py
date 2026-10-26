"""Measure complete real analysis responses; never benchmark mocks or health checks."""
import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--payload', type=Path, required=True, help='Analysis JSON request')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--target-ms', type=float, default=800)
    parser.add_argument('--output', type=Path, default=Path('output/benchmark.json'))
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.timeout <= 0 or args.target_ms <= 0:
        parser.error('runs, timeout and target must be positive; warmup must be nonnegative')
    payload = args.payload.read_bytes()
    json.loads(payload)
    samples, warmup = [], []
    for index in range(args.warmup + args.runs):
        started = time.perf_counter()
        request = Request(args.url.rstrip('/') + '/api/analyze', data=payload,
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
            if response.status != 200 or not result.get('candidates'):
                raise RuntimeError('Analysis did not return successful candidate results')
        elapsed = (time.perf_counter() - started) * 1000
        (warmup if index < args.warmup else samples).append(elapsed)
        print(f'{"warmup" if index < args.warmup else "measured"}: {elapsed:.2f} ms', flush=True)
    report = {
        'endpoint': args.url.rstrip('/') + '/api/analyze',
        'measured_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'payload_sha256': hashlib.sha256(payload).hexdigest(),
        'description': 'Sequential complete HTTP responses, including local model inference. Warmups excluded from percentiles.',
        'runs': args.runs, 'warmup_ms': warmup, 'samples_ms': samples,
        'p50_ms': statistics.median(samples),
        'p95_ms': sorted(samples)[math.ceil(.95 * len(samples)) - 1],
        'max_ms': max(samples), 'target_ms': args.target_ms,
        'all_measured_under_target': all(value < args.target_ms for value in samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
