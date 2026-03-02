"""Unit tests for release-task request model defaults and compatibility flags."""

import unittest

from acestep.api.http.release_task_models import GenerateMusicRequest
from acestep.constants import DEFAULT_DIT_INSTRUCTION


class ReleaseTaskModelsTests(unittest.TestCase):
    """Behavior tests for release-task request model schema defaults."""

    def test_generate_music_request_preserves_legacy_defaults(self):
        """Model should expose same default values used by existing clients."""

        req = GenerateMusicRequest()
        self.assertEqual("", req.prompt)
        self.assertEqual("text2music", req.task_type)
        self.assertEqual("mp3", req.audio_format)
        self.assertEqual(DEFAULT_DIT_INSTRUCTION, req.instruction)
        self.assertTrue(req.use_random_seed)


if __name__ == "__main__":
    unittest.main()
