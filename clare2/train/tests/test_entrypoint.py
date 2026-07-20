from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
import unittest


ENTRYPOINT = pathlib.Path(__file__).parents[1] / "entrypoint.sh"


class TrainingEntrypointTests(unittest.TestCase):
    def _run(self, state: dict | None = None, *, override: bool = False) -> bool:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            state_path = root / "lifecycle.json"
            marker_path = root / "training-started"
            if state is not None:
                state_path.write_text(json.dumps(state), encoding="utf-8")
            environment = {
                **os.environ,
                "CLARE2_LIFECYCLE_STATE_PATH": str(state_path),
                "CLARE2_TRAIN_AUTHORIZED": "1" if override else "0",
            }
            subprocess.run(
                ["/bin/bash", ENTRYPOINT, "/usr/bin/touch", marker_path],
                check=True,
                env=environment,
            )
            return marker_path.exists()

    def test_plain_compose_start_is_inert_without_lifecycle_state(self):
        self.assertFalse(self._run())

    def test_plain_compose_start_is_inert_while_idle(self):
        self.assertFalse(self._run({"phase": "idle"}))

    def test_stale_start_request_is_inert_outside_training_transition(self):
        self.assertFalse(self._run({"phase": "failed", "trainer_start_requested": True}))

    def test_policy_start_request_runs_training_command(self):
        self.assertTrue(
            self._run({"phase": "starting_training", "trainer_start_requested": True})
        )

    def test_explicit_dream_training_override_runs_training_command(self):
        self.assertTrue(self._run({"phase": "training"}, override=True))


if __name__ == "__main__":
    unittest.main()
