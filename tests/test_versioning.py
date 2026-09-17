from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from tools.check_release_readiness import check


ROOT = Path(__file__).resolve().parents[1]


class VersionConsistencyTest(unittest.TestCase):
    def project_version(self) -> str:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        match = re.search(
            r"^project\(lw\.PPOCR\.C VERSION ([0-9]+\.[0-9]+\.[0-9]+) LANGUAGES C\)$",
            cmake,
            re.MULTILINE,
        )
        self.assertIsNotNone(match)
        return match.group(1)

    def test_product_metadata_uses_the_cmake_version(self) -> None:
        version = self.project_version()
        self.assertEqual(version, "1.0.0")

        assembly = (
            ROOT / "examples" / "csharp-winforms" / "Properties" / "AssemblyInfo.cs"
        ).read_text(encoding="utf-8-sig")
        self.assertIn(f'AssemblyVersion("{version}.0")', assembly)
        self.assertIn(f'AssemblyFileVersion("{version}.0")', assembly)

        for relative in (
            "android/demo/build.gradle.kts",
            "android/demo-java/build.gradle.kts",
        ):
            gradle = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("versionCode = 2", gradle)
            self.assertIn(f'versionName = "{version}-preview.1"', gradle)

        sbom = json.loads((ROOT / "sbom.cdx.json").read_text(encoding="utf-8"))
        component = sbom["metadata"]["component"]
        self.assertEqual(component["version"], version)
        self.assertEqual(component["purl"], f"pkg:generic/lw.PPOCR.C@{version}")
        self.assertEqual(component["bom-ref"], component["purl"])
        dependency_refs = {item["ref"] for item in sbom["dependencies"]}
        self.assertIn(component["bom-ref"], dependency_refs)

    def test_android_workflow_does_not_request_removed_tools_package(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "android.yml").read_text(
            encoding="utf-8"
        )
        setup_start = workflow.index("uses: android-actions/setup-android@v3")
        pinned_start = workflow.index("- name: Install pinned SDK components")
        setup_block = workflow[setup_start:pinned_start]
        self.assertIn('packages: "platform-tools"', setup_block)
        self.assertNotIn('packages: "tools platform-tools"', setup_block)
        self.assertNotIn('"tools"', setup_block)

    def test_runtime_contract_snapshot_matches_sources(self) -> None:
        snapshot = json.loads(
            (ROOT / "abi" / "runtime-contract-v1.json").read_text(encoding="utf-8")
        )
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        version = self.project_version()
        self.assertEqual(snapshot["schema_version"], 1)
        self.assertEqual(snapshot["product_version"], version)
        wasm_match = re.search(
            r'set\(LW_WASM_HOST_ABI_VERSION "([0-9]+)" CACHE STRING', cmake
        )
        lwm_match = re.search(
            r'set\(LW_LWM_FORMAT_VERSION "([0-9]+\.[0-9]+)" CACHE STRING', cmake
        )
        self.assertIsNotNone(wasm_match)
        self.assertIsNotNone(lwm_match)
        assert wasm_match is not None
        assert lwm_match is not None
        self.assertEqual(snapshot["wasm_host_abi_version"], int(wasm_match.group(1)))
        self.assertEqual(
            snapshot["wasm_host_abi_manifest"], "abi/web-abi-v1-candidate.json"
        )
        self.assertTrue((ROOT / snapshot["wasm_host_abi_manifest"]).is_file())
        self.assertEqual(snapshot["lwm_format_version"], lwm_match.group(1))

        candidate = json.loads(
            (ROOT / "abi" / "c-abi-v1-candidate.json").read_text(encoding="utf-8")
        )
        self.assertEqual(snapshot["c_abi"]["version"], candidate["abi_version"])
        self.assertEqual(snapshot["c_abi"]["status"], candidate["status"])
        self.assertEqual(
            snapshot["c_abi"]["candidate_manifest"], "abi/c-abi-v1-candidate.json"
        )
        self.assertEqual(snapshot["c_abi"]["layout_manifest"], candidate["layout_manifest"])
        self.assertEqual(snapshot["lwm_layout_manifest"], candidate["lwm_layout_manifest"])
        self.assertEqual(snapshot["c_abi"]["symbol_allowlist"], candidate["stable_symbols"])
        self.assertTrue((ROOT / snapshot["orientation_contract"]).is_file())
        self.assertTrue((ROOT / snapshot["lwm_layout_manifest"]).is_file())
        lwm_layout = json.loads((ROOT / "abi" / "lwm-v0.1-layout.json").read_text(encoding="utf-8"))
        self.assertEqual(lwm_layout["format"], "LWM")
        self.assertEqual(
            f'{lwm_layout["format_major"]}.{lwm_layout["format_minor"]}',
            snapshot["lwm_format_version"],
        )

    def test_stable_release_scope_is_explicit(self) -> None:
        scope = json.loads(
            (ROOT / "ci" / "stable-release-scope.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(scope["schema_version"], 1)
        self.assertEqual(scope["release_version"], "1.0.0")
        self.assertEqual(scope["status"], "approved")
        self.assertEqual(scope["lwm_policy"], "internal-preview")
        self.assertEqual(scope["stable_models"], ["tiny"])
        self.assertEqual(scope["preview_only_models"], ["small", "medium"])

    def test_stable_gate_only_requires_models_in_stable_scope(self) -> None:
        report = check("stable", "1.0.0")
        self.assertNotIn("stable release scope is not approved", report["blockers"])
        self.assertNotIn(
            "analysis-only model variants remain: medium, small",
            report["blockers"],
        )

    def test_preview_tools_and_release_example_share_the_version_base(self) -> None:
        version = self.project_version()
        packager = (ROOT / "tools" / "package_ppocrv6_runtime.py").read_text(
            encoding="utf-8"
        )
        package_doc = (ROOT / "docs" / "package.md").read_text(encoding="utf-8")
        release_notes = (ROOT / "docs" / "release-notes-v1.0.0.md").read_text(
            encoding="utf-8"
        )
        web_doc = (ROOT / "docs" / "web-sdk.md").read_text(encoding="utf-8")
        node_packager = (ROOT / "tools" / "package_node_wasm.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(f'DEFAULT_RUNTIME_VERSION = "{version}"', packager)
        self.assertIn(f'DEFAULT_MINIMUM_RUNTIME_VERSION = "{version}"', packager)
        self.assertIn(f"`v{version}` archive is the first ABI-frozen stable release", package_doc)
        self.assertIn(f"git tag -a v{version}", package_doc)
        self.assertNotIn(f"git tag -s v{version}", package_doc)
        self.assertIn("gh attestation verify", package_doc)
        self.assertIn("# lw.PPOCR.C v1.0.0", release_notes)
        self.assertIn("Tiny DET LWM", release_notes)
        self.assertIn("frozen manifest checksums", release_notes)
        self.assertIn(f'for example "{version}"', web_doc)
        self.assertIn("frozen contract", node_packager)

    def test_preview_release_documentation_matches_supported_outputs(self) -> None:
        version = self.project_version()
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readme_zh = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
        platform_matrix = (ROOT / "docs" / "platform-matrix.md").read_text(
            encoding="utf-8"
        )
        c_api_doc = (ROOT / "docs" / "c-api.md").read_text(encoding="utf-8")
        abi_candidate_doc = (ROOT / "docs" / "c-abi-v1-candidate.md").read_text(
            encoding="utf-8"
        )
        java_readme = (ROOT / "examples" / "java-jni" / "README.md").read_text(
            encoding="utf-8"
        )
        java_readme_zh = (
            ROOT / "examples" / "java-jni" / "README.zh-CN.md"
        ).read_text(encoding="utf-8")

        self.assertIn(f"## Current stable release: v{version}", readme)
        self.assertIn(f"## 当前稳定版：v{version}", readme_zh)
        for token in ("Tiny", "Small", "Medium", "android-arm64.aar"):
            self.assertIn(token, readme)
        for token in ("Tiny", "Small", "Medium", "android-arm64.aar"):
            self.assertIn(token, readme_zh)
        self.assertIn("Do not mix binaries", readme)
        self.assertIn(f"`v{version}` stable package carries this contract", c_api_doc)
        self.assertIn(f"`v{version}` stable package", abi_candidate_doc)
        self.assertIn("不要混用不同 Release", readme_zh)

        for artifact in (
            "lw-ppocr-java-jni-windows-x64",
            "lw-ppocr-java-jni-linux-x64",
            "lw-ppocr-java-jni-macos-arm64",
        ):
            self.assertIn(artifact, java_readme)
            self.assertIn(artifact, java_readme_zh)
        self.assertIn("Desktop Java/JNI macOS ARM64", platform_matrix)


if __name__ == "__main__":
    unittest.main()
