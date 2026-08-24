"""以只读方式发现 Zotero storage 目录中的 PDF。"""

import os
from pathlib import Path
from typing import List

from app.retrieval.models import DiscoveredPDF, DiscoveryReport


_TEMP_SUFFIXES = {".tmp", ".part", ".download", ".crdownload"}
_HTML_SUFFIXES = {".html", ".htm"}


def _is_hidden_or_temporary(path: Path) -> bool:
    name = path.name
    return (
        name.startswith(".")
        or name.startswith("~")
        or name.startswith("._")
        or name.endswith("~")
        or path.suffix.lower() in _TEMP_SUFFIXES
    )


class ZoteroPDFDiscovery:
    """发现 PDF 附件，整个过程不写入或修改 Zotero。"""

    def __init__(self, storage_path: Path | str):
        self.storage_path = Path(storage_path).expanduser()

    def discover(self) -> DiscoveryReport:
        root = self.storage_path
        report = DiscoveryReport(storage_path=root)
        if not root.exists() or not root.is_dir():
            return report

        item_directories = [
            path
            for path in root.iterdir()
            if path.is_dir() and not path.is_symlink() and not _is_hidden_or_temporary(path)
        ]

        discovered: List[DiscoveredPDF] = []
        for item_dir in sorted(item_directories, key=lambda path: path.name):
            item_pdf_count = 0
            for current_root, dir_names, file_names in os.walk(item_dir, followlinks=False):
                dir_names[:] = [
                    name
                    for name in dir_names
                    if not _is_hidden_or_temporary(Path(name))
                    and not (Path(current_root) / name).is_symlink()
                ]
                report.scanned_directory_count += 1

                for file_name in file_names:
                    path = Path(current_root) / file_name
                    if _is_hidden_or_temporary(path) or path.is_symlink():
                        report.skipped_hidden_or_temporary_count += 1
                        continue
                    suffix = path.suffix.lower()
                    if suffix in _HTML_SUFFIXES:
                        report.skipped_html_snapshot_count += 1
                        continue
                    if suffix != ".pdf" or not path.is_file():
                        continue
                    discovered.append(
                        DiscoveredPDF(
                            path=path,
                            zotero_storage_key=item_dir.name,
                        )
                    )
                    item_pdf_count += 1

            if item_pdf_count == 0:
                report.skipped_no_pdf_directory_count += 1

        # Zotero 通常把附件放在 Item 子目录中，这里也兼容 storage 根目录下的 PDF。
        for path in sorted(root.iterdir(), key=lambda candidate: candidate.name):
            if (
                path.is_file()
                and not path.is_symlink()
                and not _is_hidden_or_temporary(path)
                and path.suffix.lower() == ".pdf"
            ):
                discovered.append(
                    DiscoveredPDF(path=path, zotero_storage_key=path.parent.name)
                )

        report.discovered_pdfs = sorted(
            discovered,
            key=lambda item: str(item.path).lower(),
        )
        return report
