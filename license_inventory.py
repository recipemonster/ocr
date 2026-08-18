import argparse
from dataclasses import dataclass
from importlib import metadata
import json
from pathlib import Path
import re
import shutil


LEGAL_FILE_PREFIXES = ("COPYING", "LICENSE", "NOTICE")


@dataclass(frozen=True, order=True)
class LicenseEntry:
    name: str
    version: str
    license_expression: str
    files: tuple[str, ...] = ()


def safe_name(value):
    normalized = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-.")
    return normalized or "unknown"


def declared_license(distribution):
    expression = distribution.metadata.get("License-Expression", "").strip()
    if expression:
        return expression
    legacy = distribution.metadata.get("License", "").strip()
    if legacy and legacy.upper() != "UNKNOWN":
        return legacy
    classifiers = distribution.metadata.get_all("Classifier", [])
    licenses = sorted(
        classifier.removeprefix("License :: ")
        for classifier in classifiers
        if classifier.startswith("License :: ")
    )
    return ", ".join(licenses) if licenses else "NOASSERTION"


def is_legal_file(path):
    name = Path(path).name.upper()
    return any(name.startswith(prefix) for prefix in LEGAL_FILE_PREFIXES)


def copy_distribution_licenses(distribution, destination):
    copied = []
    package_directory = destination / "third-party" / safe_name(distribution.metadata["Name"])
    for relative_path in sorted(distribution.files or [], key=str):
        if not is_legal_file(relative_path):
            continue
        source = Path(distribution.locate_file(relative_path))
        if not source.is_file():
            continue
        target = package_directory / Path(relative_path).name
        suffix = 2
        while target.exists() and target.read_bytes() != source.read_bytes():
            target = package_directory / f"{Path(relative_path).stem}-{suffix}{Path(relative_path).suffix}"
            suffix += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        copied.append(str(target.relative_to(destination)))
    return tuple(sorted(set(copied)))


def collect_python_licenses(destination):
    entries = []
    for distribution in sorted(metadata.distributions(), key=lambda item: item.metadata["Name"].lower()):
        entries.append(
            LicenseEntry(
                name=distribution.metadata["Name"],
                version=distribution.version,
                license_expression=declared_license(distribution),
                files=copy_distribution_licenses(distribution, destination),
            )
        )
    return entries


def package_purl(package):
    for reference in package.get("externalRefs", []):
        locator = reference.get("referenceLocator", "")
        if locator.startswith("pkg:apk/"):
            return locator
    return ""


def collect_spdx_licenses(directory):
    unique = set()
    for path in sorted(directory.glob("*.spdx.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for package in document.get("packages", []):
            purl = package_purl(package)
            if not purl or "origin=" in purl:
                continue
            unique.add(
                LicenseEntry(
                    name=package.get("name", "unknown"),
                    version=package.get("versionInfo", "unknown"),
                    license_expression=package.get("licenseDeclared", "NOASSERTION"),
                )
            )
    return sorted(unique)


def write_notice(entries, destination, source_description):
    destination.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Third-party notices",
        "",
        f"Generated from {source_description}.",
        "",
        "| Package | Version | License | Included files |",
        "| --- | --- | --- | --- |",
    ]
    for entry in entries:
        files = ", ".join(f"`{path}`" for path in entry.files) or "See package metadata"
        lines.append(f"| {entry.name} | {entry.version} | {entry.license_expression} | {files} |")
    lines.append("")
    (destination / "THIRD_PARTY_NOTICES.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spdx-directory", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.spdx_directory:
        entries = collect_spdx_licenses(args.spdx_directory)
        source = f"SPDX package manifests in {args.spdx_directory}"
    else:
        entries = collect_python_licenses(args.output)
        source = "the installed Python distribution metadata"
    write_notice(entries, args.output, source)


if __name__ == "__main__":
    main()
