import unittest
from web.compiled_diagnostics import is_compiled_status


class CompiledDiagnosticsTest(unittest.TestCase):
    def test_success_markers(self):
        self.assertTrue(is_compiled_status('LW_WASM_COMPILED_REC widths=5/5'))
        self.assertTrue(is_compiled_status('LW_WASM_COMPILED_REC ctc=simd128'))

    def test_errors_are_not_status(self):
        for text in ('RuntimeError: unreachable',
                     'LW_WASM_COMPILED_REC widths=5/5 error',
                     'LW_WASM_COMPILED_CLS layout=nhwc ops=101 unsupported=1 arena_bytes=0 packed_bytes=0'):
            self.assertFalse(is_compiled_status(text))


if __name__ == '__main__':
    unittest.main()
