#!/usr/bin/env python3
"""
blast_ncbi.py — NCBI BLAST integration for Sanger Processor GUI
----------------------------------------------------------------
__version__ = "2.0.0"
__date__     = "2026-06-02"

Backend options (selectable in the dialog)
------------------------------------------
  1. URL API (built-in)
       Endpoint : https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi
       Auth     : none (free, unauthenticated)
       Deps     : stdlib only (urllib)
       Limits   : 1 req/10 s, poll ≥30 s, plain-text results
       Notes    : Original backend; always available.

  2. Biopython  (Bio.Blast.NCBIWWW + NCBIXML)
       Endpoint : same CGI endpoint, but handled by Biopython
       Auth     : none
       Deps     : pip install biopython
       Limits   : same rate limits as URL API
       Notes    : More robust parsing; returns structured Hit objects;
                  XML result format; better error handling.

  3. NCBI Datasets REST API  (newer JSON-based API)
       Endpoint : https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi
                  (uses FORMAT_TYPE=JSON2 — same CGI, JSON output)
       Auth     : NCBI API key optional (raises rate limit 3→10 req/s)
       Deps     : stdlib only
       Notes    : Returns JSON; easier to parse programmatically;
                  API key can be set in NCBI account settings and
                  passed as &api_key=YOUR_KEY in the query string.

PyQt5 compatibility
-------------------
  Tested with PyQt5 5.15.x (macOS, Windows, Linux).
  No PyQt6 / PySide6 imports used.
"""

__version__ = "2.0.0"
__date__     = "2026-06-02"

import re
import time
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional, List, Dict

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QLineEdit, QProgressBar, QGroupBox,
    QTableWidget, QTableWidgetItem, QMessageBox, QApplication,
    QSplitter, QWidget, QCheckBox, QTabWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BLAST_CGI_URL   = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
POLL_INTERVAL_S = 30
MAX_WAIT_S      = 300
USER_AGENT      = f"SangerProcessorGUI/{__version__} (blast_ncbi.py)"

DATABASES = [
    ("nt",                  "nt — All nucleotide (broadest)"),
    ("refseq_rna",          "refseq_rna — RefSeq mRNA"),
    ("refseq_genomic",      "refseq_genomic — RefSeq genomes"),
    ("16S_ribosomal_RNA",   "16S rRNA — Microbial ID"),
    ("ITS_RefSeq_Fungi",    "ITS RefSeq — Fungal ID"),
    ("pdbnt",               "pdbnt — PDB nucleotide"),
    ("nr",                  "nr — Non-redundant protein (blastp)"),
    ("refseq_protein",      "refseq_protein — RefSeq protein (blastp)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _post(params: dict, extra_headers: dict = None) -> str:
    data    = urllib.parse.urlencode(params).encode()
    headers = {"User-Agent": USER_AGENT}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(BLAST_CGI_URL, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _get(params: dict, extra_headers: dict = None) -> str:
    url     = BLAST_CGI_URL + "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": USER_AGENT}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Backend 1 — URL API  (stdlib, plain text)
# ─────────────────────────────────────────────────────────────────────────────

def urlapi_submit(sequence: str, program: str, database: str,
                  email: str, megablast: bool,
                  api_key: str = "") -> str:
    """Submit via URL API; return RID."""
    if not sequence.startswith(">"):
        sequence = ">query\n" + sequence
    params = {
        "CMD":         "Put",
        "PROGRAM":     program,
        "DATABASE":    database,
        "QUERY":       sequence,
        "FORMAT_TYPE": "Text",
        "EMAIL":       email,
        "TOOL":        "SangerProcessorGUI",
    }
    if program == "blastn" and megablast:
        params["MEGABLAST"] = "on"
        params["WORD_SIZE"] = "28"
    if api_key:
        params["api_key"] = api_key
    body  = _post(params)
    match = re.search(r'RID\s*=\s*([A-Z0-9]+)', body)
    if not match:
        raise RuntimeError(
            "Could not find RID in BLAST submission response.\n"
            "Snippet:\n" + body[:500])
    return match.group(1)


def urlapi_poll(rid: str, api_key: str = "") -> str:
    """Return 'WAITING', 'READY', or 'FAILED'."""
    params = {"CMD": "Get", "FORMAT_OBJECT": "SearchInfo", "RID": rid}
    if api_key:
        params["api_key"] = api_key
    body = _get(params)
    if "Status=WAITING" in body:               return "WAITING"
    if "Status=READY"   in body:               return "READY"
    if "Status=FAILED"  in body:               return "FAILED"
    if "ThereAreHits=yes" in body:             return "READY"
    return "UNKNOWN"


def urlapi_retrieve(rid: str, max_hits: int = 20,
                    api_key: str = "") -> str:
    """Download plain-text results."""
    params = {
        "CMD":          "Get",
        "RID":          rid,
        "FORMAT_TYPE":  "Text",
        "DESCRIPTIONS": str(max_hits),
        "ALIGNMENTS":   str(max_hits),
        "HITLIST_SIZE": str(max_hits),
    }
    if api_key:
        params["api_key"] = api_key
    body = _get(params)
    if ("No significant similarity found" in body
            or "Query=" in body
            or "Sequences producing" in body
            or "<BlastOutput>" in body):
        return body
    raise RuntimeError("Unexpected result format.\nSnippet:\n" + body[:500])


# ─────────────────────────────────────────────────────────────────────────────
# Backend 2 — Biopython  (Bio.Blast.NCBIWWW + NCBIXML)
# ─────────────────────────────────────────────────────────────────────────────

def biopython_available() -> bool:
    try:
        import Bio  # noqa: F401
        return True
    except ImportError:
        return False


def biopython_blast(sequence: str, program: str, database: str,
                    email: str, megablast: bool,
                    max_hits: int = 20,
                    progress_cb=None) -> tuple:
    """
    Run BLAST via Biopython.

    Returns
    -------
    (rid, result_text, hits_list)
        rid         : str  — NCBI RID
        result_text : str  — XML string (raw Biopython output)
        hits_list   : list of dicts (parsed)
    """
    from Bio.Blast import NCBIWWW, NCBIXML
    from Bio import Entrez
    Entrez.email = email

    if not sequence.startswith(">"):
        sequence = ">query\n" + sequence

    kwargs = dict(
        program   = program,
        database  = database,
        sequence  = sequence,
        hitlist_size = max_hits,
        format_type  = "XML",
    )
    if program == "blastn" and megablast:
        kwargs["megablast"] = "T"

    if progress_cb:
        progress_cb("Submitting via Biopython (NCBIWWW)…")

    result_handle = NCBIWWW.qblast(**kwargs)
    xml_str       = result_handle.read()
    result_handle.close()

    if progress_cb:
        progress_cb("Parsing XML results…")

    from io import StringIO
    blast_records = list(NCBIXML.parse(StringIO(xml_str)))

    hits = []
    for record in blast_records:
        for i, alignment in enumerate(record.alignments):
            hsp = alignment.hsps[0]
            identity_pct = int(100 * hsp.identities / hsp.align_length)
            hits.append({
                "rank":            i + 1,
                "accession":       alignment.accession,
                "description":     alignment.title[:120],
                "score":           hsp.score,
                "evalue":          f"{hsp.expect:.2e}",
                "identity_pct":    identity_pct,
                "query_cover_pct": int(100 * (hsp.query_end - hsp.query_start)
                                       / record.query_length)
                                   if record.query_length else None,
                "align_length":    hsp.align_length,
                "gaps":            hsp.gaps,
            })

    # Extract RID from the XML (Biopython embeds it)
    rid_match = re.search(r'<RID>(.*?)</RID>', xml_str)
    rid = rid_match.group(1).strip() if rid_match else "unknown"

    return rid, xml_str, hits


# ─────────────────────────────────────────────────────────────────────────────
# Backend 3 — NCBI JSON API  (FORMAT_TYPE=JSON2, optional api_key)
# ─────────────────────────────────────────────────────────────────────────────

def jsonapi_submit(sequence: str, program: str, database: str,
                   email: str, megablast: bool,
                   api_key: str = "") -> str:
    """Submit via JSON2 format; return RID."""
    if not sequence.startswith(">"):
        sequence = ">query\n" + sequence
    params = {
        "CMD":         "Put",
        "PROGRAM":     program,
        "DATABASE":    database,
        "QUERY":       sequence,
        "FORMAT_TYPE": "JSON2",
        "EMAIL":       email,
        "TOOL":        "SangerProcessorGUI",
    }
    if program == "blastn" and megablast:
        params["MEGABLAST"] = "on"
        params["WORD_SIZE"] = "28"
    if api_key:
        params["api_key"] = api_key
    body  = _post(params)
    match = re.search(r'RID\s*=\s*([A-Z0-9]+)', body)
    if not match:
        raise RuntimeError(
            "Could not find RID in JSON API submission.\nSnippet:\n" + body[:500])
    return match.group(1)


def jsonapi_retrieve(rid: str, max_hits: int = 20,
                     api_key: str = "") -> tuple:
    """
    Download JSON2 results.

    Returns
    -------
    (raw_json_str, hits_list)
    """
    params = {
        "CMD":          "Get",
        "RID":          rid,
        "FORMAT_TYPE":  "JSON2",
        "HITLIST_SIZE": str(max_hits),
        "DESCRIPTIONS": str(max_hits),
        "ALIGNMENTS":   str(max_hits),
    }
    if api_key:
        params["api_key"] = api_key
    body = _get(params)

    # JSON2 wraps the result; extract the JSON block
    json_match = re.search(r'(\{.*\})', body, re.DOTALL)
    if not json_match:
        # May already be pure JSON
        json_str = body.strip()
    else:
        json_str = json_match.group(1)

    hits = []
    try:
        data = json.loads(json_str)
        # Navigate NCBI JSON2 structure
        report = (data.get("BlastOutput2", [{}])[0]
                      .get("report", {})
                      .get("results", {})
                      .get("search", {}))
        for i, hit in enumerate(report.get("hits", [])):
            desc  = hit.get("description", [{}])[0]
            hsps  = hit.get("hsps", [{}])
            best  = hsps[0] if hsps else {}
            aln   = best.get("align_len", 1)
            ident = best.get("identity", 0)
            hits.append({
                "rank":            i + 1,
                "accession":       desc.get("accession", "?"),
                "description":     desc.get("title", "")[:120],
                "score":           best.get("bit_score", 0),
                "evalue":          f"{best.get('evalue', 0):.2e}",
                "identity_pct":    int(100 * ident / aln) if aln else None,
                "query_cover_pct": None,
                "align_length":    aln,
                "gaps":            best.get("gaps", 0),
            })
    except (json.JSONDecodeError, KeyError, IndexError):
        # Return raw body as text if parsing fails
        pass

    return json_str, hits


# ─────────────────────────────────────────────────────────────────────────────
# Shared plain-text result parser  (for URL API backend)
# ─────────────────────────────────────────────────────────────────────────────

def parse_text_results(text: str) -> List[Dict]:
    hits      = []
    in_table  = False
    rank      = 0
    for line in text.splitlines():
        if "Sequences producing significant alignments" in line:
            in_table = True; continue
        if in_table:
            line = line.strip()
            if not line:
                if rank > 0: break
                continue
            parts = line.split()
            if len(parts) < 3: continue
            accession = parts[0]
            try:
                score  = float(parts[-2])
                evalue = parts[-1]
                desc   = " ".join(parts[1:-2])
            except (ValueError, IndexError):
                continue
            rank += 1
            hits.append({
                "rank":            rank,
                "accession":       accession,
                "description":     desc,
                "score":           score,
                "evalue":          evalue,
                "identity_pct":    None,
                "query_cover_pct": None,
            })

    identity_vals = re.findall(
        r'Identities\s*=\s*\d+/\d+\s*\((\d+)%\)', text)
    cover_vals = re.findall(
        r'Query\s+Cover\s*[=:]\s*(\d+)%', text, flags=re.IGNORECASE)
    for i, h in enumerate(hits):
        h["identity_pct"]    = int(identity_vals[i]) if i < len(identity_vals) else None
        h["query_cover_pct"] = int(cover_vals[i])    if i < len(cover_vals)    else None
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# QThread worker  (handles all three backends)
# ─────────────────────────────────────────────────────────────────────────────

class BlastWorker(QThread):
    status_update = pyqtSignal(str)
    progress_pct  = pyqtSignal(int)
    # (backend_used, RID, result_text, hits_list)
    finished      = pyqtSignal(str, str, str, list)
    error         = pyqtSignal(str)

    def __init__(self, sequence: str, program: str, database: str,
                 email: str, megablast: bool, max_hits: int,
                 backend: str = "url_api", api_key: str = ""):
        super().__init__()
        self.sequence  = sequence
        self.program   = program
        self.database  = database
        self.email     = email
        self.megablast = megablast
        self.max_hits  = max_hits
        self.backend   = backend   # 'url_api' | 'biopython' | 'json_api'
        self.api_key   = api_key
        self._abort    = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            if self.backend == "biopython":
                self._run_biopython()
            elif self.backend == "json_api":
                self._run_json_api()
            else:
                self._run_url_api()
        except urllib.error.URLError as e:
            self.error.emit(
                f"Network error — check your internet connection.\n\n{e}")
        except Exception as e:
            import traceback
            self.error.emit(f"BLAST error ({self.backend}):\n{e}\n\n"
                            + traceback.format_exc())

    # ── URL API ───────────────────────────────────────────────────────────────
    def _run_url_api(self):
        self.status_update.emit("Submitting via URL API…")
        self.progress_pct.emit(5)
        rid = urlapi_submit(self.sequence, self.program, self.database,
                            self.email, self.megablast, self.api_key)
        self.status_update.emit(f"Submitted  RID={rid}  — polling every {POLL_INTERVAL_S}s…")
        self.progress_pct.emit(15)

        elapsed = 0
        while elapsed < MAX_WAIT_S:
            if self._abort:
                self.error.emit("Cancelled by user."); return
            time.sleep(POLL_INTERVAL_S)
            elapsed += POLL_INTERVAL_S
            status   = urlapi_poll(rid, self.api_key)
            pct      = min(15 + int(70 * elapsed / MAX_WAIT_S), 85)
            self.progress_pct.emit(pct)
            self.status_update.emit(
                f"RID={rid}  status={status}  elapsed={elapsed}s")
            if status == "READY":   break
            if status == "FAILED":
                self.error.emit(f"BLAST job failed on NCBI servers (RID={rid})."); return
        else:
            self.error.emit(
                f"Timed out after {MAX_WAIT_S}s (RID={rid}).\n"
                f"Retrieve manually: {BLAST_CGI_URL}?CMD=Get&RID={rid}"); return

        self.status_update.emit(f"Downloading results (RID={rid})…")
        self.progress_pct.emit(90)
        text = urlapi_retrieve(rid, self.max_hits, self.api_key)
        hits = parse_text_results(text)
        self.progress_pct.emit(100)
        self.status_update.emit("Done.")
        self.finished.emit("URL API", rid, text, hits)

    # ── Biopython ─────────────────────────────────────────────────────────────
    def _run_biopython(self):
        if not biopython_available():
            self.error.emit(
                "Biopython is not installed.\n\n"
                "Install with:\n    pip install biopython\n\n"
                "Then restart the application."); return
        self.status_update.emit("Submitting via Biopython (this may take 30–90 s)…")
        self.progress_pct.emit(10)

        def _cb(msg):
            self.status_update.emit(msg)
            self.progress_pct.emit(50)

        rid, xml_str, hits = biopython_blast(
            self.sequence, self.program, self.database,
            self.email, self.megablast, self.max_hits, _cb)

        self.progress_pct.emit(100)
        self.status_update.emit(f"Done.  RID={rid}  ({len(hits)} hits)")
        self.finished.emit("Biopython", rid, xml_str, hits)

    # ── JSON API ──────────────────────────────────────────────────────────────
    def _run_json_api(self):
        self.status_update.emit("Submitting via JSON API…")
        self.progress_pct.emit(5)
        rid = jsonapi_submit(self.sequence, self.program, self.database,
                             self.email, self.megablast, self.api_key)
        self.status_update.emit(f"Submitted  RID={rid}  — polling every {POLL_INTERVAL_S}s…")
        self.progress_pct.emit(15)

        elapsed = 0
        while elapsed < MAX_WAIT_S:
            if self._abort:
                self.error.emit("Cancelled by user."); return
            time.sleep(POLL_INTERVAL_S)
            elapsed += POLL_INTERVAL_S
            # Poll uses the same status check as URL API
            status = urlapi_poll(rid, self.api_key)
            pct    = min(15 + int(70 * elapsed / MAX_WAIT_S), 85)
            self.progress_pct.emit(pct)
            self.status_update.emit(
                f"RID={rid}  status={status}  elapsed={elapsed}s")
            if status == "READY":   break
            if status == "FAILED":
                self.error.emit(f"BLAST job failed on NCBI servers (RID={rid})."); return
        else:
            self.error.emit(f"Timed out after {MAX_WAIT_S}s (RID={rid})."); return

        self.status_update.emit(f"Downloading JSON results (RID={rid})…")
        self.progress_pct.emit(90)
        json_str, hits = jsonapi_retrieve(rid, self.max_hits, self.api_key)
        self.progress_pct.emit(100)
        self.status_update.emit("Done.")
        self.finished.emit("JSON API", rid, json_str, hits)


# ─────────────────────────────────────────────────────────────────────────────
# Results dialog
# ─────────────────────────────────────────────────────────────────────────────

class BlastResultsDialog(QDialog):
    """
    Non-modal dialog that submits a BLAST search and displays results.
    Supports all three backends selectable at runtime.
    PyQt5 5.15 compatible.
    """

    def __init__(self, sequence: str,
                 quality_stats: Optional[dict] = None,
                 parent=None):
        super().__init__(parent)
        self.sequence      = sequence
        self.quality_stats = quality_stats or {}
        self._worker       = None
        self._rid          = None
        self._result_text  = ""
        self._hits         = []

        self.setWindowTitle(
            f"NCBI BLAST — Sanger Processor  (blast_ncbi v{__version__})")
        self.resize(960, 720)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Settings ──────────────────────────────────────────────────────────
        settings = QGroupBox("Search Settings")
        sg = QHBoxLayout(settings)

        sg.addWidget(QLabel("Backend:"))
        self.combo_backend = QComboBox()
        self.combo_backend.addItem("URL API (stdlib, no deps)",  "url_api")
        self.combo_backend.addItem("Biopython (pip install biopython)", "biopython")
        self.combo_backend.addItem("JSON API (structured output)", "json_api")
        self.combo_backend.setCurrentIndex(0)
        self.combo_backend.setToolTip(
            "URL API    — stdlib only; plain-text results; always available.\n"
            "Biopython  — structured Hit objects; XML parsing; pip install biopython.\n"
            "JSON API   — same CGI endpoint but returns JSON2; no extra deps;\n"
            "             easier to parse; supports NCBI API key for higher rate limits.")
        self.combo_backend.currentIndexChanged.connect(self._on_backend_changed)
        sg.addWidget(self.combo_backend)

        sg.addWidget(QLabel("Program:"))
        self.combo_program = QComboBox()
        self.combo_program.addItems(["blastn", "blastp"])
        sg.addWidget(self.combo_program)

        sg.addWidget(QLabel("Database:"))
        self.combo_db = QComboBox()
        for key, label in DATABASES:
            self.combo_db.addItem(label, key)
        sg.addWidget(self.combo_db)

        sg.addWidget(QLabel("Max hits:"))
        self.combo_hits = QComboBox()
        self.combo_hits.addItems(["10", "20", "50", "100"])
        self.combo_hits.setCurrentText("20")
        sg.addWidget(self.combo_hits)

        self.btn_megablast = QPushButton("MegaBLAST: ON")
        self.btn_megablast.setCheckable(True)
        self.btn_megablast.setChecked(True)
        self.btn_megablast.toggled.connect(
            lambda on: self.btn_megablast.setText(
                "MegaBLAST: ON" if on else "MegaBLAST: OFF"))
        sg.addWidget(self.btn_megablast)
        layout.addWidget(settings)

        # ── Credentials ───────────────────────────────────────────────────────
        cred = QGroupBox("Credentials")
        cg   = QHBoxLayout(cred)

        cg.addWidget(QLabel("Email (required):"))
        self.edit_email = QLineEdit()
        self.edit_email.setPlaceholderText("you@institution.ac.uk")
        cg.addWidget(self.edit_email)

        cg.addWidget(QLabel("NCBI API key (optional):"))
        self.edit_apikey = QLineEdit()
        self.edit_apikey.setPlaceholderText(
            "Paste key from https://www.ncbi.nlm.nih.gov/account/")
        self.edit_apikey.setToolTip(
            "Free NCBI API key raises the rate limit from 3 to 10 requests/second.\n"
            "Get one at: https://www.ncbi.nlm.nih.gov/account/\n"
            "Used by URL API and JSON API backends only.")
        self.edit_apikey.setEchoMode(QLineEdit.Password)
        cg.addWidget(self.edit_apikey)

        self.cb_show_key = QCheckBox("Show key")
        self.cb_show_key.toggled.connect(
            lambda on: self.edit_apikey.setEchoMode(
                QLineEdit.Normal if on else QLineEdit.Password))
        cg.addWidget(self.cb_show_key)
        layout.addWidget(cred)

        # ── Backend info label ────────────────────────────────────────────────
        self.lbl_backend_info = QLabel()
        self.lbl_backend_info.setStyleSheet("color:#555;font-style:italic;font-size:10px;")
        self.lbl_backend_info.setWordWrap(True)
        layout.addWidget(self.lbl_backend_info)
        self._on_backend_changed()   # populate label

        # ── Sequence info ─────────────────────────────────────────────────────
        qs = self.quality_stats
        seq_lbl = QLabel(
            f"Sequence: {len(self.sequence)} bases  |  "
            f"Mean Q: {qs.get('mean_phred', 0):.1f}  |  "
            f"Q≥20: {qs.get('pct_q20', 0):.1f}%  |  "
            f"Est. BLAST identity: {qs.get('blast_est_identity', 0):.1f}%"
        )
        seq_lbl.setStyleSheet("font-family:monospace;")
        layout.addWidget(seq_lbl)

        self.edit_seq = QTextEdit()
        self.edit_seq.setPlainText(self.sequence)
        self.edit_seq.setMaximumHeight(80)
        self.edit_seq.setToolTip(
            "Edit if you want to BLAST a different window or the reverse complement.")
        layout.addWidget(self.edit_seq)

        # ── Progress ──────────────────────────────────────────────────────────
        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setMinimumWidth(420)
        prog_row.addWidget(self.progress_bar)
        prog_row.addWidget(self.lbl_status)
        layout.addLayout(prog_row)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.btn_run = QPushButton("🔍  Submit BLAST Search")
        self.btn_run.setStyleSheet(
            "font-weight:bold;background-color:#1565C0;color:white;padding:6px;")
        self.btn_run.clicked.connect(self.run_blast)
        btn_row.addWidget(self.btn_run)

        self.btn_abort = QPushButton("■  Cancel")
        self.btn_abort.setEnabled(False)
        self.btn_abort.setStyleSheet(
            "font-weight:bold;background-color:#f44336;color:white;padding:5px;")
        self.btn_abort.clicked.connect(self.abort_blast)
        btn_row.addWidget(self.btn_abort)

        self.btn_open_ncbi = QPushButton("🌐  Open in Browser")
        self.btn_open_ncbi.setEnabled(False)
        self.btn_open_ncbi.clicked.connect(self._open_ncbi)
        btn_row.addWidget(self.btn_open_ncbi)

        self.btn_copy_seq = QPushButton("📋  Copy Sequence")
        self.btn_copy_seq.clicked.connect(
            lambda: QApplication.clipboard().setText(
                self.edit_seq.toPlainText().strip()))
        btn_row.addWidget(self.btn_copy_seq)

        self.btn_copy_results = QPushButton("📋  Copy Results")
        self.btn_copy_results.setEnabled(False)
        self.btn_copy_results.clicked.connect(
            lambda: QApplication.clipboard().setText(self._result_text))
        btn_row.addWidget(self.btn_copy_results)

        layout.addLayout(btn_row)

        # ── Results ───────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Vertical)

        # Hit summary table
        self.hit_table = QTableWidget(0, 7)
        self.hit_table.setHorizontalHeaderLabels(
            ["Rank", "Accession", "Description",
             "Score", "E-value", "Identity %", "Query Cover %"])
        for col, w in enumerate([40, 110, 330, 65, 80, 80, 90]):
            self.hit_table.setColumnWidth(col, w)
        self.hit_table.setMaximumHeight(200)
        splitter.addWidget(self.hit_table)

        # Raw result text
        self.text_results = QTextEdit()
        self.text_results.setReadOnly(True)
        mono = QFont("Courier")
        mono.setPointSize(9)
        self.text_results.setFont(mono)
        self.text_results.setPlaceholderText(
            "BLAST results appear here after the search completes.\n\n"
            "Typical wait: 20–90 seconds.\n\n"
            f"blast_ncbi.py  v{__version__}  ({__date__})\n"
            "Backends: URL API | Biopython | JSON API")
        splitter.addWidget(self.text_results)
        layout.addWidget(splitter, 1)

        # Version footer
        footer = QLabel(
            f"blast_ncbi.py  v{__version__}  ({__date__})  —  "
            f"PyQt5 {self._pyqt_version()}  —  "
            f"Biopython: {'✓ installed' if biopython_available() else '✗ not installed'}")
        footer.setStyleSheet("color:#888;font-size:9px;")
        layout.addWidget(footer)

    @staticmethod
    def _pyqt_version() -> str:
        try:
            from PyQt5.QtCore import PYQT_VERSION_STR
            return PYQT_VERSION_STR
        except Exception:
            return "?"

    def _on_backend_changed(self):
        backend = self.combo_backend.currentData()
        info = {
            "url_api": (
                "URL API — stdlib only, no extra installs.  "
                "Returns plain text.  Rate: 3 req/s (10/s with API key).  "
                "BLAST+ version: whatever NCBI is running (2.16.x as of mid-2026)."
            ),
            "biopython": (
                "Biopython — requires:  pip install biopython  "
                "Returns structured XML parsed into Hit objects with full HSP details.  "
                f"Installed: {'Yes ✓' if biopython_available() else 'No ✗ — install first'}."
            ),
            "json_api": (
                "JSON API — stdlib only.  Same CGI endpoint but requests FORMAT_TYPE=JSON2.  "
                "Returns structured JSON; easier to parse programmatically.  "
                "Supports NCBI API key for higher rate limits."
            ),
        }.get(backend, "")
        if hasattr(self, 'lbl_backend_info'):
            self.lbl_backend_info.setText(info)

    # ─────────────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────────────

    def run_blast(self):
        sequence = self.edit_seq.toPlainText().strip()
        if not sequence:
            QMessageBox.warning(self, "No Sequence", "Enter a sequence to BLAST.")
            return
        email = self.edit_email.text().strip()
        if not email or "@" not in email:
            QMessageBox.warning(self, "Email Required",
                "NCBI's Terms of Service require a valid contact email address.")
            return

        backend = self.combo_backend.currentData()
        if backend == "biopython" and not biopython_available():
            QMessageBox.critical(self, "Biopython Not Installed",
                "Install Biopython first:\n\n    pip install biopython\n\n"
                "Then restart the application, or choose a different backend.")
            return

        self.btn_run.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.btn_open_ncbi.setEnabled(False)
        self.btn_copy_results.setEnabled(False)
        self.hit_table.setRowCount(0)
        self.text_results.clear()
        self.progress_bar.setValue(0)
        self._rid = None

        db_key = self.combo_db.currentData()

        self._worker = BlastWorker(
            sequence  = sequence,
            program   = self.combo_program.currentText(),
            database  = db_key,
            email     = email,
            megablast = self.btn_megablast.isChecked(),
            max_hits  = int(self.combo_hits.currentText()),
            backend   = backend,
            api_key   = self.edit_apikey.text().strip(),
        )
        self._worker.status_update.connect(self.lbl_status.setText)
        self._worker.progress_pct.connect(self.progress_bar.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def abort_blast(self):
        if self._worker:
            self._worker.abort()
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.lbl_status.setText("Cancelled.")

    def _on_finished(self, backend_used: str, rid: str,
                     result_text: str, hits: list):
        self._rid         = rid
        self._result_text = result_text
        self._hits        = hits
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.btn_open_ncbi.setEnabled(bool(rid and rid != "unknown"))
        self.btn_copy_results.setEnabled(True)
        self.lbl_status.setText(
            f"Done.  Backend={backend_used}  RID={rid}  Hits={len(hits)}")
        self.text_results.setPlainText(result_text)
        self._populate_hit_table(hits)

        if not hits and "No significant similarity found" in result_text:
            QMessageBox.information(self, "No Hits",
                "BLAST returned no significant hits.\n\n"
                "Suggestions:\n"
                "  • Try the reverse complement\n"
                "  • Turn MegaBLAST OFF\n"
                "  • Select a broader database (nt)\n"
                "  • Check the BLAST window quality in the main GUI")

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.lbl_status.setText("Error — see dialog.")
        QMessageBox.critical(self, "BLAST Error", msg)

    def _populate_hit_table(self, hits: list):
        self.hit_table.setRowCount(len(hits))
        for row, h in enumerate(hits):
            id_pct  = h.get("identity_pct")
            cov_pct = h.get("query_cover_pct")
            for col, val in enumerate([
                str(h.get("rank", row + 1)),
                h.get("accession", ""),
                h.get("description", ""),
                str(h.get("score", "")),
                str(h.get("evalue", "")),
                f"{id_pct}%"  if id_pct  is not None else "—",
                f"{cov_pct}%" if cov_pct is not None else "—",
            ]):
                item = QTableWidgetItem(val)
                # Colour-code by identity
                if id_pct is not None:
                    if id_pct >= 99:
                        item.setBackground(QColor(200, 255, 200))
                    elif id_pct >= 95:
                        item.setBackground(QColor(255, 255, 200))
                    elif id_pct < 80:
                        item.setBackground(QColor(255, 220, 220))
                self.hit_table.setItem(row, col, item)

    def _open_ncbi(self):
        if not self._rid: return
        import webbrowser
        webbrowser.open(
            f"https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
            f"?CMD=Get&RID={self._rid}")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — call from GUI.py
# ─────────────────────────────────────────────────────────────────────────────

def open_blast_dialog(parent, sequence: str,
                      quality_stats: Optional[dict] = None):
    """
    Open the BLAST dialog non-modally.

    Usage in SangerGUI:
        from blast_ncbi import open_blast_dialog
        open_blast_dialog(self, seq, self.result.get('quality_stats'))
    """
    dlg = BlastResultsDialog(sequence, quality_stats, parent=parent)
    dlg.show()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    TEST_SEQ = (
        "ATGAAAGCAATTTTCGTACTGAAAGGTTTTGTTGGTTTTCTTCTGGCATCAGCGTTTGCC"
        "ATCGCAACTCTTGGAGCTTTTGTACAAGCTGTGGATGATGTGGTTTCTGCCCGGTTGTTG"
    )
    print(f"blast_ncbi.py  v{__version__}  ({__date__})")
    print(f"Biopython available: {biopython_available()}")
    print("\nSubmitting test URL API search…")
    try:
        rid = urlapi_submit(TEST_SEQ, "blastn", "nt",
                            "test@example.com", megablast=True)
        print(f"RID: {rid}")
        for i in range(MAX_WAIT_S // POLL_INTERVAL_S):
            time.sleep(POLL_INTERVAL_S)
            status = urlapi_poll(rid)
            print(f"  [{(i+1)*POLL_INTERVAL_S}s] {status}")
            if status == "READY":
                text = urlapi_retrieve(rid, max_hits=5)
                hits = parse_text_results(text)
                print(f"Top {len(hits)} hits:")
                for h in hits:
                    print(f"  [{h['rank']}] {h['accession']}  "
                          f"score={h['score']}  e={h['evalue']}  "
                          f"id={h.get('identity_pct')}%")
                break
            if status == "FAILED":
                print("Job failed."); sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")