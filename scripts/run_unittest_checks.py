#!/usr/bin/env python3
"""Run a required unittest suite and reject skipped or empty validation."""
import argparse
import json
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--discover', type=Path)
    selection.add_argument('--module')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Refusing to overwrite unittest evidence')
    sys.path.insert(0, str(Path.cwd()))
    loader = unittest.defaultTestLoader
    suite = loader.discover(str(args.discover)) if args.discover else loader.loadTestsFromName(args.module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.wasSuccessful() and result.testsRun > 0 and not result.skipped
    record = {'status': 'passed' if passed else 'failed', 'tests_run': result.testsRun,
              'skipped': [{'test': str(test), 'reason': reason} for test, reason in result.skipped],
              'failures': len(result.failures), 'errors': len(result.errors),
              'expected_failures': len(result.expectedFailures), 'unexpected_successes': len(result.unexpectedSuccesses)}
    # Expected failures are unresolved required behavior and cannot be a release pass.
    if result.expectedFailures:
        passed = False
        record['status'] = 'failed'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    print(json.dumps(record))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
