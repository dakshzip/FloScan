"""Read-only P06 review probes. All mutations affect temporary synthetic input."""

import json
import runpy
import tempfile
from pathlib import Path

from floscan.capture import stray
from floscan.capture.sync import associate


def main():
    helpers = runpy.run_path("tests/integration/test_stray.py")
    with tempfile.TemporaryDirectory(prefix="floscan-p06-review-") as temporary:
        session = helpers["make_session"](Path(temporary) / "session")

        def inspect(label):
            try:
                result = stray.inspect_session(stray.open_session(session))
                print(label, result.status)
                try:
                    json.dumps(stray.report(result), allow_nan=False)
                    print(label, "strict JSON: valid")
                except ValueError as error:
                    print(label, "strict JSON:", type(error).__name__, str(error))
            except Exception as error:
                print(label, type(error).__name__, str(error))

        inspect("baseline")
        confidence = session / "confidence/000000.png"
        original = confidence.read_bytes()
        confidence.write_bytes(b"bad PNG")
        inspect("corrupt confidence at sampled pair endpoint")
        confidence.write_bytes(original)

        imu = session / "imu.csv"
        original = imu.read_text()
        header = original.splitlines()[0]
        imu.write_text(header + "\n")
        inspect("header-only IMU")
        imu.write_text(header + "\nnan,nan,nan,nan,nan,nan,nan\n")
        inspect("non-finite IMU")
        imu.write_text(original)

        stream = [0, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15, 0.17, 0.2, 0.22]
        sensor = [10 + value for value in stream]
        try:
            result = associate(stream, sensor, 0.004, 0, 0.99, 0.05)
            print("zero start offset", result.status)
        except Exception as error:
            print("zero start offset", type(error).__name__, str(error))


if __name__ == "__main__":
    main()
