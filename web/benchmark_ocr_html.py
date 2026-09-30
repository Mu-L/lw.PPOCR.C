"""Alternating same-browser full OCR A/B; timing is informational."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from playwright.sync_api import sync_playwright
from compiled_diagnostics import assert_compiled_widths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--golden', type=Path, required=True)
    parser.add_argument('--candidate-golden', type=Path,
                        help='Explicit reviewed candidate contract; baseline still uses --golden')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--iterations', type=int, default=9)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--interleave', action='store_true',
                        help='Alternate individual OCR calls between two idle contexts to reduce temporal drift')
    parser.add_argument('--baseline-widths', type=int)
    parser.add_argument('--candidate-widths', type=int)
    parser.add_argument('--diagnostic-text-differences', action='store_true',
                        help='Report candidate text differences without treating timing as a validated gain')
    args = parser.parse_args()
    if min(args.iterations, args.rounds) < 1:
        parser.error('iterations and rounds must be positive')
    runs = {'baseline': [], 'candidate': []}
    golden = json.loads(args.golden.read_text(encoding='utf-8'))
    goldens = {'baseline': golden, 'candidate': golden}
    if args.candidate_golden:
        goldens['candidate'] = json.loads(args.candidate_golden.read_text(encoding='utf-8'))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for round_id in range(args.rounds):
                order = ('baseline', 'candidate') if round_id % 2 == 0 else ('candidate', 'baseline')
                if args.interleave:
                    contexts, pages = {}, {}
                    times = {name: [] for name in order}
                    hashes = {name: [] for name in order}
                    results = {}
                    ready_ms, cls_reconfigure_ms, logs, first_ms = {}, {}, {}, {}
                    try:
                        for name in order:
                            contexts[name] = browser.new_context()
                            page = contexts[name].new_page()
                            page.set_default_timeout(180000)
                            logs[name] = []
                            page.on('console', lambda msg, log=logs[name]: log.append(msg.text))
                            started = time.perf_counter()
                            page.goto(getattr(args, name).resolve().as_uri(), timeout=180000)
                            page.evaluate('() => window.lwPpocrDemo.ready()')
                            ready_ms[name] = (time.perf_counter() - started) * 1000
                            started = time.perf_counter()
                            page.locator('#use-cls').check()
                            page.wait_for_function("() => !document.querySelector('#use-cls').disabled")
                            cls_reconfigure_ms[name] = (time.perf_counter() - started) * 1000
                            page.locator('#file').set_input_files(str(args.sample.resolve()))
                            page.wait_for_function('() => window.__lwOcrTest.snapshot().prepared')
                            pages[name] = page
                        for index in range(args.iterations + 3):
                            # No overlapping inference; change the first engine every call.
                            call_order = order if index % 2 == 0 else order[::-1]
                            for name in call_order:
                                result = pages[name].evaluate('''async () => {
                                  const start = performance.now();
                                  await window.lwPpocrDemo.recognize();
                                  return {ms: performance.now() - start,
                                    text: window.lwPpocrDemo.getPlainText(),
                                    lines: window.lwPpocrDemo.getResult().lines.length,
                                    heap: window.__lwOcrTest.snapshot().heapBytes};
                                }''')
                                hashes[name].append(hashlib.sha256(result['text'].encode()).hexdigest())
                                if (result['lines'] != goldens[name]['expected_line_count'] or
                                    (hashes[name][-1] != goldens[name]['expected_text_sha256'] and
                                     not (args.diagnostic_text_differences and name == 'candidate'))):
                                    raise AssertionError('OCR golden contract mismatch: ' + json.dumps(result, ensure_ascii=True))
                                if index == 0:
                                    first_ms[name] = result['ms']
                                if index >= 3:
                                    times[name].append(result['ms'])
                                results[name] = result
                        for name in order:
                            if len(set(hashes[name])) != 1:
                                raise AssertionError('non-deterministic OCR')
                            run = {'median_ms': statistics.median(times[name]), 'times_ms': times[name],
                                   'text_sha256': hashes[name][0], 'heap_bytes': results[name]['heap'],
                                   'ready_ms': ready_ms[name], 'cls_reconfigure_ms': cls_reconfigure_ms[name],
                                   'first_ocr_ms': first_ms[name], 'text_lines': results[name]['text'].split('\n')}
                            expected_widths = getattr(args, name + '_widths')
                            if expected_widths is not None:
                                run['compiled_widths'] = assert_compiled_widths('\n'.join(logs[name]), expected_widths)
                            runs[name].append(run)
                            print(json.dumps({'round': round_id + 1, 'label': name, **run}), flush=True)
                    finally:
                        for context in contexts.values():
                            context.close()
                    continue
                for name in order:
                    context = browser.new_context()
                    try:
                        page = context.new_page()
                        page.set_default_timeout(180000)
                        log = []
                        page.on('console', lambda msg: log.append(msg.text))
                        started = time.perf_counter()
                        page.goto(getattr(args, name).resolve().as_uri(), timeout=180000)
                        page.evaluate('() => window.lwPpocrDemo.ready()')
                        ready_ms = (time.perf_counter() - started) * 1000
                        started = time.perf_counter()
                        page.locator('#use-cls').check()
                        page.wait_for_function("() => !document.querySelector('#use-cls').disabled")
                        cls_reconfigure_ms = (time.perf_counter() - started) * 1000
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
                            if (result['lines'] != goldens[name]['expected_line_count'] or
                                (hashes[-1] != goldens[name]['expected_text_sha256'] and
                                 not (args.diagnostic_text_differences and name == 'candidate'))):
                                raise AssertionError('OCR golden contract mismatch: ' + json.dumps(result, ensure_ascii=True))
                            if index == 0:
                                first_ms = result['ms']
                            if index >= 3:
                                times.append(result['ms'])
                        if len(set(hashes)) != 1:
                            raise AssertionError('non-deterministic OCR')
                        run = {'median_ms': statistics.median(times), 'times_ms': times,
                               'text_sha256': hashes[0], 'heap_bytes': result['heap'],
                               'ready_ms': ready_ms, 'cls_reconfigure_ms': cls_reconfigure_ms,
                               'first_ocr_ms': first_ms, 'text_lines': result['text'].split('\n')}
                        expected_widths = getattr(args, name + '_widths')
                        if expected_widths is not None:
                            run['compiled_widths'] = assert_compiled_widths('\n'.join(log), expected_widths)
                        runs[name].append(run)
                        print(json.dumps({'round': round_id + 1, 'label': name, **run}), flush=True)
                    finally:
                        context.close()
        finally:
            browser.close()
    text_pass = all(r['text_sha256'] == goldens[name]['expected_text_sha256']
                    for name, group in runs.items() for r in group)
    if not text_pass and not args.diagnostic_text_differences:
        raise AssertionError('baseline/candidate text mismatch')
    report = {'schema_version': 1, 'paired_speedup': statistics.median(
        b['median_ms'] / c['median_ms'] for b, c in zip(runs['baseline'], runs['candidate'])),
        'sample_sha256': hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        'use_cls': True, 'warmup': 3, 'iterations': args.iterations,
        'interleaved_calls': args.interleave,
        'text_contract_pass': text_pass,
        'texts_identical': len({r['text_sha256'] for group in runs.values() for r in group}) == 1,
        'expected_text_sha256': {name: value['expected_text_sha256'] for name, value in goldens.items()},
        'timing_validated': text_pass,
        'browser_version': browser.version, 'runs': runs}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
