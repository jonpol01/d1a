"""The PR labeling job's questions and document builder, extracted from the production script (personal-pr-label.py,
snapshot 2026-10-03) so records are built exactly as production asks. Update both together."""
import re

TYPES = (
    "type/bug",
    "type/docs",
    "type/feature",
    "type/perf",
    "type/refactor",
    "type/security",
    "type/test",
)


BLASTS = (
    "review:blast-contained",
    "review:blast-moderate",
    "review:blast-broad",
    "review:blast-massive",
)


SEVS = ("P0", "P1", "P2", "P3", "P4")


D1A_QUESTIONS = {
    "type": {
        "type": "choice",
        "instructions": "Primary change type from files and body, not the title prefix.",
        "criteria": {
            "type/bug": "Defect / incorrect behavior",
            "type/docs": "Documentation only",
            "type/feature": "New behavior",
            "type/perf": "Performance",
            "type/refactor": "No intended behavior change",
            "type/security": "Auth, secrets, or permissions",
            "type/test": "Tests or CI",
        },
    },
    "blast": {
        "type": "choice",
        "instructions": "How far a mistake in this PR spreads in production.",
        "criteria": {
            "review:blast-contained": "One module",
            "review:blast-moderate": "One subsystem",
            "review:blast-broad": "Shared helper or config",
            "review:blast-massive": "Auth, permissions, or all paths",
        },
    },
    "sev": {
        "type": "choice",
        "instructions": "How serious the problem this PR addresses is — not the risk of merging the diff as-is. P0 = the bug/outage being fixed is drop-everything (data loss, a security hole being closed, crash loop). P1 = major break, no workaround. P2 = degraded, workaround exists. P3 = cosmetic / nice-to-have. P4 = best-effort, no promise. Dependabot lockfile bumps without CVE are P4. Docs-only is P4. CI/test infra that can block main is P2. A PR that itself adds .env/secrets is classified by a separate rule, not this scale.",
        "criteria": {
            "P0": "Drop everything — data loss, security, crash loop",
            "P1": "Major break, no workaround",
            "P2": "Degraded, workaround exists",
            "P3": "Cosmetic or nice-to-have",
            "P4": "Best-effort, no promise",
        },
    },
}


def _pr_state(pr: dict, files: list[dict]) -> str:
    title = (pr.get("title") or "")[:240]
    author = ((pr.get("user") or {}).get("login") or "")
    body = re.sub(r"<[^>]+>", " ", pr.get("body") or "")
    body = re.sub(r"\s+", " ", body).strip()[:3500]
    add = int(pr.get("additions") or 0)
    dele = int(pr.get("deletions") or 0)
    nfiles = int(pr.get("changed_files") or 0) or len(files)
    lines = [
        f"title: {title}",
        f"author: {author}",
        f"stats: +{add}/-{dele} files={nfiles}",
        "body:",
        body or "(empty)",
        "files:",
    ]
    for f in files[:40]:
        lines.append(
            f"- {f['status']} {f['filename']} +{f['additions']}/-{f['deletions']}"
        )
    if not files:
        lines.append("- (none listed)")
    return "\n".join(lines)
