"""Alternating same-browser full OCR A/B; timing is informational."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--golden', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--iterations', type=int, default=9)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if min(args.iterations, args.rounds) < 1:
        parser.error('iterations and rounds must be positive')
    runs = {'baseline': [], 'candidate': []}
    golden = json.loads(args.golden.read_text(encoding='utf-8'))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for round_id in range(args.rounds):
                order = ('baseline', 'candidate') if round_id % 2 == 0 else ('candidate', 'baseline')
                for name in order:
                    context = browser.new_context()
                    try:
                        page = context.new_page()
                        page.set_default_timeout(180000)
                        page.goto(getattr(args, name).resolve().as_uri(), timeout=180000)
                        page.evaluate('() => window.lwPpocrDemo.ready()')
                        page.locator('#use-cls').check()
                        page.wait_for_function("() => !document.querySelector('#use-cls').disabled")
                        page.locator('#file').set_input_files(str(args.sample.resolve()))
                        page.wait_for_function('() => window.__lwOcrTest.snapshot().prepared')
                        times, hashes = [], []
                        for index in range(args.iterations + 3):
                            result = page.evaluate('''async () => {
                              const start = performance.now();
                              await window.lwPpocrDemo.recognize();
                              return {ms: performance.now() - start,
                                text: window.lwPpocrDemo.getPlainText(),
                                lines: window.lwPpocrDemo.getResult().lines.length,
                                heap: window.__lwOcrTest.snapshot().heapBytes};
                            }''')
                            hashes.append(hashlib.sha256(result['text'].encode()).hexdigest())
                            if (hashes[-1] != golden['expected_text_sha256'] or
                                    result['lines'] != golden['expected_line_count']):
                                raise AssertionError('OCR golden contract mismatch: ' + json.dumps(result, ensure_ascii=True))
                            if index >= 3:
                                times.append(result['ms'])
                        if len(set(hashes)) != 1:
                            raise AssertionError('non-deterministic OCR')
                        run = {'median_ms': statistics.median(times), 'times_ms': times,
                               'text_sha256': hashes[0], 'heap_bytes': result['heap']}
                        runs[name].append(run)
                        print(json.dumps({'round': round_id + 1, 'label': name, **run}), flush=True)
                    finally:
                        context.close()
        finally:
            browser.close()
    if len({r['text_sha256'] for group in runs.values() for r in group}) != 1:
        raise AssertionError('baseline/candidate text mismatch')
    report = {'schema_version': 1, 'paired_speedup': statistics.median(
        b['median_ms'] / c['median_ms'] for b, c in zip(runs['baseline'], runs['candidate'])),
        'sample_sha256': hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        'use_cls': True, 'warmup': 3, 'iterations': args.iterations,
        'browser_version': browser.version, 'runs': runs}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
