from email.message import Message
from pathlib import Path
import tempfile
import unittest

import license_inventory


class FakeDistribution:
    def __init__(self, root, name, version, license_expression):
        self.root = root
        self.version = version
        self.metadata = Message()
        self.metadata["Name"] = name
        self.metadata["License-Expression"] = license_expression
        self.files = [Path("demo.dist-info/licenses/LICENSE")]

    def locate_file(self, path):
        return self.root / path


class LicenseInventoryTest(unittest.TestCase):
    def test_python_inventory_copies_legal_files_and_writes_notice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            license_path = root / "demo.dist-info" / "licenses" / "LICENSE"
            license_path.parent.mkdir(parents=True)
            license_path.write_text("license text", encoding="utf-8")
            destination = root / "output"
            distribution = FakeDistribution(root, "Demo Package", "1.2.3", "MIT")

            entries = [
                license_inventory.LicenseEntry(
                    name=distribution.metadata["Name"],
                    version=distribution.version,
                    license_expression=license_inventory.declared_license(distribution),
                    files=license_inventory.copy_distribution_licenses(distribution, destination),
                )
            ]
            license_inventory.write_notice(entries, destination, "test metadata")

            notice = (destination / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
            copied = destination / "third-party" / "demo-package" / "LICENSE"
            self.assertIn("| Demo Package | 1.2.3 | MIT |", notice)
            self.assertEqual(copied.read_text(encoding="utf-8"), "license text")

    def test_spdx_inventory_selects_installed_apk_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "package.spdx.json"
            document.write_text(
                """{
  "packages": [
    {
      "name": "glib",
      "versionInfo": "2.0-r1",
      "licenseDeclared": "LGPL-2.1-or-later",
      "externalRefs": [{
        "referenceLocator": "pkg:apk/wolfi/glib@2.0-r1?arch=x86_64"
      }]
    },
    {
      "name": "glib-source",
      "versionInfo": "2.0",
      "licenseDeclared": "LGPL-2.1-or-later",
      "externalRefs": [{
        "referenceLocator": "pkg:github/example/glib@2.0"
      }]
    }
  ]
}
""",
                encoding="utf-8",
            )

            entries = license_inventory.collect_spdx_licenses(root)

            self.assertEqual(
                entries,
                [license_inventory.LicenseEntry("glib", "2.0-r1", "LGPL-2.1-or-later")],
            )


if __name__ == "__main__":
    unittest.main()
