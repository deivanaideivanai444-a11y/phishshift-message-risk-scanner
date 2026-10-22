"""Run tests and save an honest, machine-readable outcome summary."""

import json
import unittest
from pathlib import Path


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes.append({"test": str(test), "outcome": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcomes.append({"test": str(test), "outcome": "failed", "details": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.outcomes.append({"test": str(test), "outcome": "error", "details": self._exc_info_to_string(err, test)})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.outcomes.append({"test": str(test), "outcome": "skipped", "details": reason})


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult)
    result = runner.run(suite)
    report = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
        "outcomes": result.outcomes,
    }
    output = Path("reports/test_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Recorded test outcomes at {output.resolve()}")
    raise SystemExit(0 if result.wasSuccessful() else 1)
