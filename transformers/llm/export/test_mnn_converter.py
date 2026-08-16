import json
import tempfile
import unittest

from utils.mnn_converter import MNNConverter


class GateFoldMetadataTest(unittest.TestCase):
    @staticmethod
    def _converter(expected_ops):
        converter = MNNConverter.__new__(MNNConverter)
        converter.gate_fold_ops = set(expected_ops)
        return converter

    @staticmethod
    def _write_graph(param):
        graph = {
            "oplists": [{
                "type": "LinearAttention",
                "name": "/layers.0/self_attn/FusedLinearAttention",
                "main": param,
            }]
        }
        output = tempfile.NamedTemporaryFile(mode="w", suffix=".json")
        json.dump(graph, output)
        output.flush()
        return output

    def test_accepts_complete_gate_fold_metadata(self):
        graph = self._write_graph({
            "attn_type": "gated_delta_rule",
            "num_v_heads": 2,
            "gate_fold": True,
            "gate_coef": [-1.0, -2.0],
            "gate_bias": [0.1, 0.2],
        })
        self._converter({"/layers.0/self_attn/FusedLinearAttention"}).validate_linear_attention_gate_fold(
            graph.name)

    def test_rejects_metadata_dropped_by_stale_converter(self):
        graph = self._write_graph({
            "attn_type": "gated_delta_rule",
            "num_v_heads": 2,
        })
        with self.assertRaisesRegex(RuntimeError, "same source tree"):
            self._converter({"/layers.0/self_attn/FusedLinearAttention"}).validate_linear_attention_gate_fold(
                graph.name)

    def test_ignores_ops_that_did_not_request_gate_fold(self):
        graph = self._write_graph({
            "attn_type": "gated_delta_rule",
            "num_v_heads": 2,
        })
        self._converter(set()).validate_linear_attention_gate_fold(graph.name)


if __name__ == "__main__":
    unittest.main()
