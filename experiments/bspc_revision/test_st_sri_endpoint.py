import sys
import unittest
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import ST_SRI_Interpreter


class EndpointModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs[:, :, 0].sum(dim=1)
        return torch.stack([score, -score], dim=1)


class StSriEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        background = torch.zeros((2, 20, 1), dtype=torch.float32)
        self.interpreter = ST_SRI_Interpreter(EndpointModel(), background)
        self.sample = torch.ones((20, 1), dtype=torch.float32)

    def test_explicit_endpoint_limits_lag_support(self) -> None:
        lags_ms, synergy, redundancy = self.interpreter.scan_fast(
            self.sample,
            max_lag_ms=20,
            stride=1,
            block_size=4,
            current_endpoint=9,
            target_cls=0,
        )

        self.assertEqual(len(lags_ms), 6)
        self.assertEqual(len(synergy), 6)
        self.assertEqual(len(redundancy), 6)

    def test_rejects_endpoint_without_complete_current_block(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot support"):
            self.interpreter.scan_fast(
                self.sample,
                block_size=4,
                current_endpoint=2,
            )


if __name__ == "__main__":
    unittest.main()
