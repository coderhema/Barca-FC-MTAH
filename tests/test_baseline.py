import tempfile
import unittest

import numpy as np

import baseline


class BaselineStateTests(unittest.TestCase):
    def test_track_teams_ignores_non_team_slots(self):
        xml = """\
<Root>
  <Team id="barca" name="Barcelona"/>
  <Team id="athletic" name="Atletic Club"/>
  <Team id="ref" name="Referee"/>
  <Player teamId="barca"/>
  <Player teamId="athletic"/>
  <Player teamId="ref"/>
</Root>
"""
        with tempfile.NamedTemporaryFile("w", suffix=".xml") as fh:
            fh.write(xml)
            fh.flush()
            np.testing.assert_array_equal(
                baseline.track_teams(fh.name),
                [baseline.HOME, baseline.AWAY, baseline.NONE],
            )

    def test_pending_restart_frames_are_backfilled(self):
        ctrl = np.full(5, baseline.NONE, dtype=np.int8)
        nearest = np.array([baseline.NONE, baseline.NONE, baseline.HOME,
                            baseline.HOME, baseline.HOME], dtype=np.int8)
        dead = np.zeros(5, dtype=bool)
        sp_team = np.full(5, baseline.NONE, dtype=np.int8)
        ball_seen = np.ones(5, dtype=bool)

        result = baseline.label_half(0, 5, ctrl, nearest, dead, sp_team, ball_seen)

        np.testing.assert_array_equal(result, [baseline.HOME] * 5)

    def test_missing_ball_does_not_create_frozen_dead_ball(self):
        shape = (5, 2)
        px = np.full(shape, 0.5, dtype=np.float32)
        py = np.full(shape, 0.5, dtype=np.float32)
        ps = np.zeros(shape, dtype=np.float32)
        bx = np.full(5, np.nan, dtype=np.float32)
        by = np.full(5, np.nan, dtype=np.float32)
        ctrl = np.full(5, baseline.NONE, dtype=np.int8)

        dead = baseline.dead_mask(px, py, ps, bx, by, ctrl, [], 5)

        self.assertFalse(dead.any())


if __name__ == "__main__":
    unittest.main()
