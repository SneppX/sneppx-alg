"""Container Security — S7: SBOM, CVE scanning, A/B partitions, signed updates."""

import os
import json
import time
import tempfile
import hashlib
import subprocess
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path


def _default_sneppx_dir() -> str:
    """Return a writable base directory for SNEPPX state.

    Honours ``SX_STATE_DIR`` / ``SNEPPX_STATE_DIR`` when set, otherwise falls
    back to a per-user temp directory so the code works without root access.
    """
    env = os.environ.get("SX_STATE_DIR") or os.environ.get("SNEPPX_STATE_DIR")
    if env:
        return env
    return os.path.join(tempfile.gettempdir(), "sneppx")


def _resolve_update_dir(update_dir: Optional[str]) -> str:
    if update_dir is not None:
        return update_dir
    return os.path.join(_default_sneppx_dir(), "updates")


class ScanStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class UpdateStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    INSTALLING = "installing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class CVE:
    id: str
    severity: Severity
    cvss_score: float
    description: str
    package: str
    version: str
    fixed_version: Optional[str] = None
    references: List[str] = field(default_factory=list)
    published_date: Optional[str] = None


@dataclass
class SBOMComponent:
    name: str
    version: str
    supplier: Optional[str] = None
    description: Optional[str] = None
    licenses: List[str] = field(default_factory=list)
    hashes: Dict[str, str] = field(default_factory=dict)
    purl: Optional[str] = None  # Package URL


@dataclass
class SBOM:
    bom_format: str = "CycloneDX"
    spec_version: str = "1.5"
    serial_number: str = ""
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    components: List[SBOMComponent] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    services: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.serial_number:
            self.serial_number = (
                f"urn:uuid:{hashlib.sha256(str(time.time()).encode()).hexdigest()[:32]}"
            )

    def to_cyclonedx_json(self) -> Dict[str, Any]:
        return {
            "bomFormat": self.bom_format,
            "specVersion": self.spec_version,
            "serialNumber": self.serial_number,
            "version": self.version,
            "metadata": self.metadata,
            "components": [
                {
                    "name": c.name,
                    "version": c.version,
                    "supplier": {"name": c.supplier} if c.supplier else None,
                    "description": c.description,
                    "licenses": [{"id": l} for l in c.licenses],
                    "hashes": [{"alg": k, "content": v} for k, v in c.hashes.items()],
                    "purl": c.purl,
                }
                for c in self.components
            ],
            "dependencies": self.dependencies,
            "services": self.services,
        }


@dataclass
class ScanResult:
    scan_id: str
    image_name: str
    status: ScanStatus
    started_at: str
    completed_at: Optional[str] = None
    cves: List[CVE] = field(default_factory=list)
    sbom: Optional[SBOM] = None
    error: Optional[str] = None

    def get_severity_counts(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for cve in self.cves:
            counts[cve.severity.value] += 1
        return counts


class CVEScanner:
    """CVE vulnerability scanner with multiple backend support."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".sneppx" / "cve_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[CVE]] = {}
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self):
        cache_file = self.cache_dir / "cve_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    for pkg, cves in data.items():
                        self._cache[pkg] = [
                            CVE(
                                id=c["id"],
                                severity=Severity(c["severity"]),
                                cvss_score=c["cvss_score"],
                                description=c["description"],
                                package=c["package"],
                                version=c["version"],
                                fixed_version=c.get("fixed_version"),
                                references=c.get("references", []),
                                published_date=c.get("published_date"),
                            )
                            for c in cves
                        ]
            except Exception:
                pass

    def _save_cache(self):
        cache_file = self.cache_dir / "cve_cache.json"
        try:
            data = {}
            for pkg, cves in self._cache.items():
                data[pkg] = [
                    {
                        "id": c.id,
                        "severity": c.severity.value,
                        "cvss_score": c.cvss_score,
                        "description": c.description,
                        "package": c.package,
                        "version": c.version,
                        "fixed_version": c.fixed_version,
                        "references": c.references,
                        "published_date": c.published_date,
                    }
                    for c in cves
                ]
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def scan_package(self, package: str, version: str) -> List[CVE]:
        """Scan a single package for CVEs."""
        cache_key = f"{package}@{version}"

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Try multiple sources
        cves = []

        # Try NVD API
        try:
            cves.extend(self._scan_nvd(package, version))
        except Exception:
            pass

        # Try OSV.dev
        try:
            cves.extend(self._scan_osv(package, version))
        except Exception:
            pass

        # Try GitHub Advisory Database
        try:
            cves.extend(self._scan_github_advisory(package, version))
        except Exception:
            pass

        with self._lock:
            self._cache[cache_key] = cves
            self._save_cache()

        return cves

    def _scan_nvd(self, package: str, version: str) -> List[CVE]:
        """Scan using NVD API v2 (public, rate-limited without API key)."""
        import urllib.request
        import urllib.parse

        cves: List[CVE] = []
        try:
            query = urllib.parse.quote(package)
            url = (
                "https://services.nvd.nist.gov/rest/json/cves/2.0"
                "?keywordSearch=%s&resultsPerPage=20" % query
            )
            req = urllib.request.Request(url, headers={"User-Agent": "sneppx-alg"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return cves

        for item in data.get("vulnerabilities", []):
            cve = item.get("cve", {})
            cve_id = cve.get("id", "")
            descriptions = cve.get("descriptions", [])
            desc = ""
            for d in descriptions:
                if d.get("lang") == "en":
                    desc = d.get("value", "")
                    break
            if not desc and descriptions:
                desc = descriptions[0].get("value", "")

            # CVSS score
            metrics = cve.get("metrics", {})
            cvss_score = 0.0
            for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                if key in metrics and metrics[key]:
                    cvss_score = (
                        metrics[key][0].get("cvssData", {}).get("baseScore", 0.0)
                    )
                    break

            if cvss_score >= 9.0:
                severity = Severity.CRITICAL
            elif cvss_score >= 7.0:
                severity = Severity.HIGH
            elif cvss_score >= 4.0:
                severity = Severity.MEDIUM
            elif cvss_score > 0:
                severity = Severity.LOW
            else:
                severity = Severity.INFO

            refs = [r.get("url", "") for r in cve.get("references", [])]
            published = cve.get("published")

            cves.append(
                CVE(
                    id=cve_id,
                    severity=severity,
                    cvss_score=float(cvss_score),
                    description=desc,
                    package=package,
                    version=version,
                    references=refs,
                    published_date=published,
                )
            )
        return cves

    def _scan_osv(self, package: str, version: str) -> List[CVE]:
        """Scan using OSV.dev API."""
        try:
            import urllib.request
            import urllib.parse

            url = "https://api.osv.dev/v1/query"
            payload = {
                "package": {"name": package},
                "version": version,
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            cves = []
            for vuln in data.get("vulns", []):
                severity = Severity.MEDIUM
                cvss_score = 5.0
                for s in vuln.get("severity", []):
                    if s.get("type") == "CVSS_V3":
                        cvss_score = s.get("score", 5.0)
                        if cvss_score >= 9.0:
                            severity = Severity.CRITICAL
                        elif cvss_score >= 7.0:
                            severity = Severity.HIGH
                        elif cvss_score >= 4.0:
                            severity = Severity.MEDIUM
                        else:
                            severity = Severity.LOW
                        break

                fixed_version = None
                for affected in vuln.get("affected", []):
                    for r in affected.get("ranges", []):
                        for event in r.get("events", []):
                            if "fixed" in event:
                                fixed_version = event["fixed"]
                                break

                cves.append(
                    CVE(
                        id=vuln.get("id", ""),
                        severity=severity,
                        cvss_score=cvss_score,
                        description=vuln.get("summary", vuln.get("details", "")),
                        package=package,
                        version=version,
                        fixed_version=fixed_version,
                        references=[
                            r.get("url", "") for r in vuln.get("references", [])
                        ],
                        published_date=vuln.get("published"),
                    )
                )
            return cves
        except Exception:
            return []

    def _scan_github_advisory(self, package: str, version: str) -> List[CVE]:
        """Scan using GitHub Advisory Database (public REST endpoint)."""
        import urllib.request
        import urllib.parse

        cves: List[CVE] = []
        try:
            eco = urllib.parse.quote("pip")
            pkg = urllib.parse.quote(package)
            url = (
                "https://api.github.com/advisories"
                "?ecosystem=%s&package=%s&per_page=20" % (eco, pkg)
            )
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "sneppx-alg",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return cves

        if not isinstance(data, list):
            data = data.get("advisories", []) if isinstance(data, dict) else []

        for adv in data:
            ghsa_id = adv.get("ghsa_id", adv.get("id", ""))
            cve_id = adv.get("cve_id") or ghsa_id
            summary = adv.get("summary", "")
            desc = adv.get("description", summary)
            severity_str = (adv.get("severity") or "medium").lower()
            severity_map = {
                "critical": Severity.CRITICAL,
                "high": Severity.HIGH,
                "medium": Severity.MEDIUM,
                "low": Severity.LOW,
            }
            severity = severity_map.get(severity_str, Severity.MEDIUM)

            cvss = adv.get("cvss") or {}
            cvss_score = float(cvss.get("score", 0.0) or 0.0)
            if cvss_score == 0.0:
                # fall back from severity
                cvss_score = {
                    "critical": 9.5,
                    "high": 7.5,
                    "medium": 5.0,
                    "low": 2.5,
                }.get(severity_str, 5.0)

            fixed_version = None
            for vuln in adv.get("vulnerabilities", []):
                for r in vuln.get("first_patched_version", {}) or {}:
                    pass
                fpv = vuln.get("first_patched_version")
                if isinstance(fpv, dict):
                    fixed_version = fpv.get("identifier")
                elif isinstance(fpv, str):
                    fixed_version = fpv

            refs = [u for u in adv.get("references", []) if isinstance(u, str)]
            published = adv.get("published_at")

            cves.append(
                CVE(
                    id=cve_id,
                    severity=severity,
                    cvss_score=cvss_score,
                    description=desc,
                    package=package,
                    version=version,
                    fixed_version=fixed_version,
                    references=refs,
                    published_date=published,
                )
            )
        return cves

    def scan_requirements(self, requirements_path: str) -> Dict[str, List[CVE]]:
        """Scan a requirements.txt or pyproject.toml file."""
        results = {}

        if requirements_path.endswith(".txt"):
            with open(requirements_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split("==")
                        if len(parts) == 2:
                            pkg, ver = parts
                            results[f"{pkg}@{ver}"] = self.scan_package(
                                pkg.strip(), ver.strip()
                            )
        elif requirements_path.endswith("pyproject.toml"):
            try:
                import tomli

                with open(requirements_path, "rb") as f:
                    data = tomli.load(f)
                deps = data.get("project", {}).get("dependencies", [])
                for dep in deps:
                    if ">=" in dep:
                        pkg, ver = dep.split(">=")
                    elif "==" in dep:
                        pkg, ver = dep.split("==")
                    else:
                        continue
                    results[f"{pkg}@{ver}"] = self.scan_package(
                        pkg.strip(), ver.strip()
                    )
            except Exception:
                pass

        return results


class SBOMGenerator:
    """Generate Software Bill of Materials (SBOM) in CycloneDX and SPDX formats."""

    def __init__(self):
        self._lock = threading.Lock()

    def generate_from_pip(self, output_path: Optional[str] = None) -> SBOM:
        """Generate SBOM from installed pip packages."""
        try:
            import pkg_resources
        except ImportError:
            return SBOM()

        sbom = SBOM()
        sbom.metadata = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tools": [{"name": "SneppX SBOM Generator", "version": "0.9.0"}],
            "component": {
                "type": "application",
                "name": "sneppx-app",
                "version": "0.9.0",
            },
        }

        for dist in pkg_resources.working_set:
            hashes = {}
            try:
                for algo in ["sha256", "sha512", "md5"]:
                    h = hashlib.new(algo)
                    for f in dist.get_metadata_lines("RECORD") or []:
                        path = f.split(",")[0]
                        full = Path(dist.location) / path
                        if full.exists():
                            h.update(full.read_bytes())
                    hashes[algo] = h.hexdigest()
            except Exception:
                pass

            licenses = []
            try:
                licenses = dist.get_metadata_lines("METADATA")
                licenses = [l for l in licenses if l.startswith("License:")]
                licenses = [l.split(":", 1)[1].strip() for l in licenses]
            except Exception:
                licenses = ["Unknown"]

            component = SBOMComponent(
                name=dist.key,
                version=dist.version,
                supplier=None,
                description=dist.get_metadata("SUMMARY") or "",
                licenses=licenses,
                hashes=hashes,
                purl=f"pkg:pypi/{dist.key}@{dist.version}",
            )
            sbom.components.append(component)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(sbom.to_cyclonedx_json(), f, indent=2)

        return sbom

    def generate_from_docker(
        self, image: str, output_path: Optional[str] = None
    ) -> SBOM:
        """Generate SBOM from Docker image (uses syft if available)."""
        sbom = SBOM()
        sbom.metadata = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tools": [{"name": "syft", "version": "unknown"}],
            "component": {"type": "container", "name": image},
        }

        try:
            # Try to use syft
            result = subprocess.run(
                ["syft", "-o", "cyclonedx-json", image],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Parse syft output into SBOM
                for comp in data.get("components", []):
                    sbom.components.append(
                        SBOMComponent(
                            name=comp.get("name", ""),
                            version=comp.get("version", ""),
                            supplier=comp.get("supplier", {}).get("name")
                            if comp.get("supplier")
                            else None,
                            description=comp.get("description", ""),
                            licenses=[
                                l.get("license", {}).get("id", "")
                                for l in comp.get("licenses", [])
                            ],
                            hashes={
                                h.get("alg", ""): h.get("content", "")
                                for h in comp.get("hashes", [])
                            },
                            purl=comp.get("purl"),
                        )
                    )
        except Exception:
            pass

        if output_path:
            with open(output_path, "w") as f:
                json.dump(sbom.to_cyclonedx_json(), f, indent=2)

        return sbom

    def to_spdx(self, sbom: SBOM) -> str:
        """Convert SBOM to SPDX tag-value format."""
        lines = [
            "SPDXVersion: SPDX-2.3",
            "DataLicense: CC0-1.0",
            "SPDXID: SPDXRef-DOCUMENT",
            f"DocumentName: {sbom.metadata.get('component', {}).get('name', 'sneppx')}",
            f"DocumentNamespace: https://sneppx.org/sbom/{sbom.serial_number}",
            f"Creator: Tool: sneppx-sbom-generator-0.9.0",
            f"Created: {sbom.metadata.get('timestamp', datetime.utcnow().isoformat())}Z",
        ]

        for i, comp in enumerate(sbom.components):
            pkg_id = f"SPDXRef-Package-{i}"
            lines.extend(
                [
                    "",
                    f"PackageName: {comp.name}",
                    f"SPDXID: {pkg_id}",
                    f"PackageVersion: {comp.version}",
                    f"PackageDownloadLocation: NOASSERTION",
                    f"FilesAnalyzed: false",
                    f"LicenseConcluded: {' OR '.join(comp.licenses) if comp.licenses else 'NOASSERTION'}",
                    f"PackageSupplier: {comp.supplier or 'NOASSERTION'}",
                    f"PackageDescription: {comp.description or 'NOASSERTION'}",
                ]
            )
            if comp.purl:
                lines.append(f"ExternalRef: {pkg_id} {comp.purl}")

        return "\n".join(lines)


class ABOOTPartition:
    """A/B partition management for atomic updates."""

    def __init__(self, root: str = "/"):
        self.root = Path(root)
        self.current = self._detect_current_slot()

    def _detect_current_slot(self) -> str:
        """Detect current A/B slot from boot params."""
        try:
            with open("/proc/cmdline", "r") as f:
                cmdline = f.read()
            for part in cmdline.split():
                if part.startswith("androidboot.slot_suffix="):
                    return part.split("=")[1]
                if part.startswith("root=") and "_a" in part:
                    return "_a"
                if part.startswith("root=") and "_b" in part:
                    return "_b"
        except Exception:
            pass
        return "_a"  # default

    @property
    def inactive_slot(self) -> str:
        return "_b" if self.current == "_a" else "_a"

    def switch_slot(self) -> bool:
        """Switch to other slot on next boot."""
        try:
            # This would use bootctl or similar on real systems
            subprocess.run(["bootctl", "set-slot", self.inactive_slot[1:]], check=True)
            return True
        except Exception:
            return False

    def get_partition_path(self, partition: str, slot: Optional[str] = None) -> Path:
        """Get partition path for given slot."""
        slot = slot or self.current
        return self.root / f"{partition}{slot}"


class SignedUpdateManager:
    """Signed update management with rollback support."""

    def __init__(
        self,
        update_dir: Optional[str] = None,
        public_key_path: Optional[str] = None,
    ):
        self.update_dir = Path(_resolve_update_dir(update_dir))
        self.update_dir.mkdir(parents=True, exist_ok=True)
        self.public_key_path = public_key_path
        self._lock = threading.Lock()

    def verify_signature(self, update_path: str, signature_path: str) -> bool:
        """Verify update package signature using Ed25519."""
        if not self.public_key_path:
            return False
        try:
            import nacl.signing
            import nacl.encoding

            with open(self.public_key_path, "rb") as f:
                pk = f.read()
            verify_key = nacl.signing.VerifyKey(pk, encoder=nacl.encoding.RawEncoder)

            with open(update_path, "rb") as f:
                data = f.read()
            with open(signature_path, "rb") as f:
                sig = f.read()

            verify_key.verify(data, sig, encoder=nacl.encoding.RawEncoder)
            return True
        except Exception:
            return False

    def stage_update(
        self,
        update_path: str,
        version: str,
        signature_path: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Stage an update for installation."""
        with self._lock:
            update_id = f"update_{version}_{int(time.time())}"
            staged_dir = self.update_dir / "staged" / update_id
            staged_dir.mkdir(parents=True, exist_ok=True)

            # Copy update
            import shutil

            dest = staged_dir / Path(update_path).name
            shutil.copy2(update_path, dest)

            # Verify signature if provided
            if signature_path and not self.verify_signature(str(dest), signature_path):
                shutil.rmtree(staged_dir)
                return False, "Signature verification failed"

            # Save metadata
            meta = {
                "update_id": update_id,
                "version": version,
                "path": str(dest),
                "staged_at": datetime.utcnow().isoformat(),
                "verified": signature_path is not None,
            }
            with open(staged_dir / "meta.json", "w") as f:
                json.dump(meta, f)

            return True, update_id

    def install_update(
        self, update_id: str, partition: ABOOTPartition
    ) -> Tuple[bool, str]:
        """Install staged update to inactive partition."""
        with self._lock:
            staged_dir = self.update_dir / "staged" / update_id
            meta_file = staged_dir / "meta.json"

            if not meta_file.exists():
                return False, "Update not found"

            with open(meta_file, "r") as f:
                meta = json.load(f)

            update_file = staged_dir / meta["path"].split("/")[-1]
            if not update_file.exists():
                return False, "Update file missing"

            # Apply update to inactive partition
            inactive = partition.inactive_slot
            part_path = partition.get_partition_path("rootfs", inactive)

            try:
                # This would use dd, flash, or similar on real systems
                subprocess.run(
                    [
                        "dd",
                        f"if={update_file}",
                        f"of={part_path}",
                        "bs=4M",
                        "status=progress",
                    ],
                    check=True,
                )

                # Switch slot
                if not partition.switch_slot():
                    return False, "Failed to switch boot slot"

                # Mark as installed
                installed_dir = self.update_dir / "installed"
                installed_dir.mkdir(exist_ok=True)
                import shutil

                shutil.move(str(staged_dir), str(installed_dir / update_id))

                return True, "Update installed successfully"
            except Exception as e:
                return False, f"Installation failed: {e}"

    def rollback(self, partition: ABOOTPartition) -> bool:
        """Rollback to previous slot."""
        return partition.switch_slot()


class ContainerSecurityManager:
    """Unified container security manager."""

    def __init__(
        self,
        update_dir: Optional[str] = None,
        public_key_path: Optional[str] = None,
        cve_cache_dir: Optional[str] = None,
    ):
        self.cve_scanner = CVEScanner(cve_cache_dir)
        self.sbom_generator = SBOMGenerator()
        self.partition_manager = ABOOTPartition()
        self.update_manager = SignedUpdateManager(update_dir, public_key_path)
        self._scan_history: List[ScanResult] = []
        self._lock = threading.Lock()

    def scan_image(
        self,
        image: str,
        generate_sbom: bool = True,
    ) -> ScanResult:
        """Scan a container image for vulnerabilities."""
        scan_id = f"scan_{image.replace('/', '_').replace(':', '_')}_{int(time.time())}"

        result = ScanResult(
            scan_id=scan_id,
            image_name=image,
            status=ScanStatus.RUNNING,
            started_at=datetime.utcnow().isoformat(),
        )

        try:
            # Generate SBOM
            sbom = None
            if generate_sbom:
                sbom = self.sbom_generator.generate_from_docker(image)

            # Scan each component
            cves = []
            for comp in sbom.components:
                comp_cves = self.cve_scanner.scan_package(comp.name, comp.version)
                cves.extend(comp_cves)

            result.cves = cves
            result.sbom = sbom
            result.status = ScanStatus.COMPLETED
            result.completed_at = datetime.utcnow().isoformat()

        except Exception as e:
            result.status = ScanStatus.FAILED
            result.error = str(e)

        with self._lock:
            self._scan_history.append(result)
            # Keep only last 100 scans
            if len(self._scan_history) > 100:
                self._scan_history = self._scan_history[-100:]

        return result

    def scan_requirements(self, requirements_path: str) -> Dict[str, List[CVE]]:
        """Scan a requirements file."""
        return self.cve_scanner.scan_requirements(requirements_path)

    def generate_sbom(
        self,
        source: str = "pip",
        output_path: Optional[str] = None,
        image: Optional[str] = None,
    ) -> SBOM:
        """Generate SBOM from various sources."""
        if source == "pip":
            return self.sbom_generator.generate_from_pip(output_path)
        elif source == "docker" and image:
            return self.sbom_generator.generate_from_docker(image, output_path)
        else:
            raise ValueError(f"Unknown source: {source}")

    def stage_update(
        self,
        update_path: str,
        version: str,
        signature_path: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Stage an update for installation."""
        return self.update_manager.stage_update(update_path, version, signature_path)

    def install_update(self, update_id: str) -> Tuple[bool, str]:
        """Install staged update."""
        return self.update_manager.install_update(update_id, self.partition_manager)

    def rollback(self) -> bool:
        """Rollback to previous slot."""
        return self.update_manager.rollback(self.partition_manager)

    def get_scan_history(self, limit: int = 50) -> List[ScanResult]:
        """Get recent scan history."""
        with self._lock:
            return self._scan_history[-limit:]

    def get_current_slot(self) -> str:
        return self.partition_manager.current

    def get_partition_info(self) -> Dict[str, Any]:
        return {
            "current_slot": self.partition_manager.current,
            "inactive_slot": self.partition_manager.inactive_slot,
        }


# Global instance
_global_container_security: Optional[ContainerSecurityManager] = None


def get_container_security(
    update_dir: Optional[str] = None,
    public_key_path: Optional[str] = None,
    cve_cache_dir: Optional[str] = None,
) -> ContainerSecurityManager:
    global _global_container_security
    if _global_container_security is None:
        _global_container_security = ContainerSecurityManager(
            update_dir, public_key_path, cve_cache_dir
        )
    return _global_container_security


def set_container_security(manager: ContainerSecurityManager):
    global _global_container_security
    _global_container_security = manager


def scan_image(image: str, generate_sbom: bool = True) -> ScanResult:
    return get_container_security().scan_image(image, generate_sbom)


def scan_requirements(path: str) -> Dict[str, List[CVE]]:
    return get_container_security().scan_requirements(path)


def generate_sbom(
    source: str = "pip", output_path: Optional[str] = None, image: Optional[str] = None
) -> SBOM:
    return get_container_security().generate_sbom(source, output_path, image)


def stage_update(
    update_path: str, version: str, signature_path: Optional[str] = None
) -> Tuple[bool, str]:
    return get_container_security().stage_update(update_path, version, signature_path)


def install_update(update_id: str) -> Tuple[bool, str]:
    return get_container_security().install_update(update_id)


def rollback_update() -> bool:
    return get_container_security().rollback()


def get_scan_history(limit: int = 50) -> List[ScanResult]:
    return get_container_security().get_scan_history(limit)


__all__ = [
    "CVE",
    "SBOMComponent",
    "SBOM",
    "ScanResult",
    "ScanStatus",
    "Severity",
    "CVEScanner",
    "SBOMGenerator",
    "ABOOTPartition",
    "SignedUpdateManager",
    "ContainerSecurityManager",
    "get_container_security",
    "set_container_security",
    "scan_image",
    "scan_requirements",
    "generate_sbom",
    "stage_update",
    "install_update",
    "rollback_update",
    "get_scan_history",
]
