import json
import unittest

from tools.run_layout_plan_snapshot import check_expected, parse_output


class LayoutPlanSnapshotTest(unittest.TestCase):
    def test_parse_output_preserves_node_dump(self) -> None:
        output = """model=rec.lwm input=[1,3,48,960]
layout_plan tensor_count=3 node_count=2 nhwc_nodes=1 nchw_nodes=1 conversions=1 islands=1 direct_input=0 width=960
nodes:
000 Conv               NCHW inputs=0,1 output=2
001 Add                NHWC inputs=2,3 output=4
"""
        parsed = parse_output(output)
        self.assertEqual(parsed["summary"]["node_count"], 2)
        self.assertEqual(parsed["nodes"][0]["op"], "Conv")
        self.assertEqual(parsed["nodes"][1]["layout"], "NHWC")
        self.assertEqual(parsed["nodes"][1]["inputs"], [2, 3])

    def test_parse_output_rejects_missing_node(self) -> None:
        output = """layout_plan tensor_count=3 node_count=2 nhwc_nodes=1 nchw_nodes=1 conversions=1 islands=1 direct_input=0 width=960
nodes:
000 Conv               NCHW inputs=0,1 output=2
"""
        with self.assertRaisesRegex(ValueError, "has 1 nodes"):
            parse_output(output)

    def test_expected_contract_is_exact(self) -> None:
        summary = {
            "tensor_count": 3,
            "node_count": 2,
            "nhwc_nodes": 1,
            "nchw_nodes": 1,
            "conversions": 1,
            "islands": 1,
            "width": 960,
        }
        expected = dict(summary)
        check_expected(summary, expected, "tiny")
        expected["islands"] = 2
        with self.assertRaisesRegex(ValueError, "tiny layout snapshot mismatch"):
            check_expected(summary, expected, "tiny")


if __name__ == "__main__":
    unittest.main()