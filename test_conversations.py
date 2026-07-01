"""
test_conversations.py
=====================
Automated test harness that:
  1. Parses ALL markdown files in D:/sample_conversations/GenAI_SampleConversations/
  2. Replays each multi-turn conversation against the live FastAPI server
  3. Compares agent output to expected: recommendations list + end_of_conversation
  4. Prints a color-coded summary report with PASS/FAIL/PARTIAL per turn

Usage (with server running):
    python test_conversations.py
    python test_conversations.py --url http://localhost:8000
    python test_conversations.py --md-dir "D:/sample_conversations/GenAI_SampleConversations"
    python test_conversations.py --file C1.md           # run single file
    python test_conversations.py --verbose              # show full agent replies
    python test_conversations.py --delay 5              # seconds between turns
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

# ── ANSI colors ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ExpectedRec:
    name: str
    url: str


@dataclass
class TurnExpected:
    turn_num: int
    user_msg: str
    has_recommendations: bool        # True -> agent MUST return recs
    expected_recs: list              # list[ExpectedRec]
    end_of_conversation: bool


@dataclass
class ConversationDef:
    file: str
    name: str
    turns: list                      # list[TurnExpected]


@dataclass
class TurnResult:
    turn_num: int
    user_msg: str
    expected: object                 # TurnExpected
    actual_reply: str
    actual_recs: list
    actual_eoc: bool
    status: str                      # "PASS" | "PARTIAL" | "FAIL" | "ERROR"
    notes: list = field(default_factory=list)


# ── Markdown parser ───────────────────────────────────────────────────────────

def _parse_table_rows(lines):
    """Parse markdown table rows into list of ExpectedRec."""
    recs = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|") or line.startswith("| #") or line.startswith("|---"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 7:
            continue
        name = cols[1].strip()
        url_raw = cols[6].strip()
        url_match = re.search(r"<(https?://[^>]+)>", url_raw)
        url = url_match.group(1) if url_match else url_raw.strip("<>")
        if name and url:
            recs.append(ExpectedRec(name=name, url=url))
    return recs


def parse_markdown(md_path):
    """Parse a GenAI sample conversation markdown into a ConversationDef."""
    text = md_path.read_text(encoding="utf-8")

    # Split into turn blocks
    turn_blocks = re.split(r"(?=### Turn \d+)", text)

    turns = []

    for block in turn_blocks:
        m = re.match(r"### Turn (\d+)", block.strip())
        if not m:
            continue
        turn_num = int(m.group(1))

        # Extract user message from blockquote
        user_lines = re.findall(r"^\s*>\s*(.+)$", block, re.MULTILINE)
        user_msg = " ".join(l.strip() for l in user_lines).strip()

        # recommendations: null -> no recs expected
        no_recs = bool(re.search(r"recommendations:\s*null", block))

        # Parse table rows
        table_lines = [l for l in block.splitlines() if l.strip().startswith("|")]
        expected_recs = _parse_table_rows(table_lines) if not no_recs else []

        # end_of_conversation
        eoc_match = re.search(r"`end_of_conversation`[:\s*]+\*\*(\w+)\*\*", block)
        eoc = eoc_match.group(1).lower() == "true" if eoc_match else False

        has_recs = not no_recs and bool(expected_recs)

        turns.append(TurnExpected(
            turn_num=turn_num,
            user_msg=user_msg,
            has_recommendations=has_recs,
            expected_recs=expected_recs,
            end_of_conversation=eoc,
        ))

    return ConversationDef(
        file=md_path.name,
        name=md_path.stem,
        turns=turns,
    )


# ── API client ────────────────────────────────────────────────────────────────

def call_chat(base_url, messages, timeout=60):
    resp = requests.post(
        f"{base_url}/chat",
        json={"messages": messages},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


# ── Evaluation ────────────────────────────────────────────────────────────────

def _normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def evaluate_turn(expected, actual):
    """Compare actual API response to the expected turn from markdown."""
    notes = []

    actual_recs  = actual.get("recommendations", [])
    actual_eoc   = actual.get("end_of_conversation", False)
    actual_reply = actual.get("reply", "")

    # Check recommendations
    rec_ok = True
    if expected.has_recommendations:
        if not actual_recs:
            notes.append("MISSING_RECS: Expected recommendations but got none")
            rec_ok = False
        else:
            actual_urls = {_normalize(r["url"]) for r in actual_recs}
            for exp_rec in expected.expected_recs:
                if _normalize(exp_rec.url) not in actual_urls:
                    notes.append(f"MISSING_REC: {exp_rec.name}")
                    rec_ok = False
    else:
        # null turn - should have no recs
        if actual_recs:
            notes.append(f"UNEXPECTED_RECS: Got {len(actual_recs)} recs when none expected")
            rec_ok = False

    # Check end_of_conversation
    eoc_ok = (actual_eoc == expected.end_of_conversation)
    if not eoc_ok:
        notes.append(
            f"EOC_MISMATCH: expected={expected.end_of_conversation} actual={actual_eoc}"
        )

    # Determine status
    if rec_ok and eoc_ok:
        status = "PASS"
    elif rec_ok or eoc_ok:
        status = "PARTIAL"
    else:
        status = "FAIL"

    return TurnResult(
        turn_num=expected.turn_num,
        user_msg=expected.user_msg,
        expected=expected,
        actual_reply=actual_reply,
        actual_recs=actual_recs,
        actual_eoc=actual_eoc,
        status=status,
        notes=notes,
    )


# ── Runner ────────────────────────────────────────────────────────────────────

def run_conversation(conv, base_url, verbose=False, delay=2.0):
    """Replay a conversation turn-by-turn and collect results."""
    history = []
    results = []

    print(f"\n{BOLD}{CYAN}{'-'*70}{RESET}")
    print(f"{BOLD}{CYAN}FILE: {conv.file}  ({len(conv.turns)} turns){RESET}")
    print(f"{CYAN}{'-'*70}{RESET}")

    for turn in conv.turns:
        if not turn.user_msg:
            continue

        history.append({"role": "user", "content": turn.user_msg})

        print(f"\n  {BOLD}Turn {turn.turn_num}{RESET}")
        short_msg = turn.user_msg[:100] + ("..." if len(turn.user_msg) > 100 else "")
        print(f"  {YELLOW}USER:{RESET} {short_msg}")

        try:
            actual = call_chat(base_url, history)
        except Exception as e:
            result = TurnResult(
                turn_num=turn.turn_num,
                user_msg=turn.user_msg,
                expected=turn,
                actual_reply="",
                actual_recs=[],
                actual_eoc=False,
                status="ERROR",
                notes=[f"API_ERROR: {e}"],
            )
            results.append(result)
            _print_turn_result(result, verbose)
            break

        result = evaluate_turn(turn, actual)
        results.append(result)

        agent_reply = actual.get("reply", "")
        if agent_reply:
            history.append({"role": "assistant", "content": agent_reply})

        _print_turn_result(result, verbose)

        if actual.get("end_of_conversation"):
            break

        if delay > 0:
            time.sleep(delay)

    return results


def _print_turn_result(result, verbose):
    if result.status == "PASS":
        color, icon = GREEN, "[PASS]"
    elif result.status == "PARTIAL":
        color, icon = YELLOW, "[PARTIAL]"
    elif result.status == "ERROR":
        color, icon = RED, "[ERROR]"
    else:
        color, icon = RED, "[FAIL]"

    print(f"  {color}{icon}{RESET}  eoc={result.actual_eoc}  recs={len(result.actual_recs)}")
    for note in result.notes:
        print(f"         >> {note}")

    if verbose:
        print(f"  {CYAN}REPLY:{RESET} {result.actual_reply[:300]}")
        if result.actual_recs:
            print(f"  {CYAN}RECS:{RESET}")
            for r in result.actual_recs:
                print(f"    * [{r.get('test_type','?')}] {r.get('name','?')}")


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(all_results):
    print(f"\n{BOLD}{'='*70}")
    print(f"  SUMMARY REPORT")
    print(f"{'='*70}{RESET}")

    total_turns = total_pass = total_partial = total_fail = total_error = 0
    rows = []

    for conv_name, results in all_results.items():
        passes   = sum(1 for r in results if r.status == "PASS")
        partials = sum(1 for r in results if r.status == "PARTIAL")
        fails    = sum(1 for r in results if r.status == "FAIL")
        errors   = sum(1 for r in results if r.status == "ERROR")
        total    = len(results)

        total_turns   += total
        total_pass    += passes
        total_partial += partials
        total_fail    += fails
        total_error   += errors

        if fails + errors == 0 and partials == 0:
            icon = f"{GREEN}OK {RESET}"
        elif fails + errors == 0:
            icon = f"{YELLOW}WRN{RESET}"
        else:
            icon = f"{RED}ERR{RESET}"

        rows.append((conv_name, icon, passes, partials, fails, errors, total))

    print(f"  {'File':<12} {'':5} {'PASS':>5} {'PART':>5} {'FAIL':>5} {'ERR':>5} {'TOTAL':>6}")
    print(f"  {'-'*12} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*6}")

    for conv_name, icon, p, pa, f, e, t in rows:
        print(f"  {conv_name:<12} {icon}   {p:>5} {pa:>5} {f:>5} {e:>5} {t:>6}")

    print(f"  {'-'*12} {'':5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*6}")
    print(f"  {'TOTAL':<12} {'':5} {total_pass:>5} {total_partial:>5} {total_fail:>5} {total_error:>5} {total_turns:>6}")

    pct = (total_pass / total_turns * 100) if total_turns else 0
    print(f"\n  {BOLD}Pass rate: {pct:.1f}%  ({total_pass}/{total_turns} turns){RESET}")

    if total_fail + total_error == 0 and total_partial == 0:
        print(f"\n  {GREEN}{BOLD}ALL TURNS PASSED!{RESET}")
    elif total_fail + total_error == 0:
        print(f"\n  {YELLOW}{BOLD}Some turns have PARTIAL matches - review above.{RESET}")
    else:
        print(f"\n  {RED}{BOLD}Some turns FAILED - review above.{RESET}")

    # Save JSON report
    report = {
        "total_turns": total_turns,
        "pass": total_pass,
        "partial": total_partial,
        "fail": total_fail,
        "error": total_error,
        "pass_rate_pct": round(pct, 2),
        "conversations": {
            conv_name: [
                {
                    "turn": r.turn_num,
                    "status": r.status,
                    "eoc_expected": r.expected.end_of_conversation,
                    "eoc_actual": r.actual_eoc,
                    "recs_expected": len(r.expected.expected_recs),
                    "recs_actual": len(r.actual_recs),
                    "notes": r.notes,
                }
                for r in results
            ]
            for conv_name, results in all_results.items()
        },
    }
    out_path = Path(__file__).parent / "test_results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  JSON report saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

DEFAULT_MD_DIR = r"D:\sample_conversations\GenAI_SampleConversations"
DEFAULT_URL    = "http://localhost:8000"


def main():
    parser = argparse.ArgumentParser(description="SHL API Conversation Test Runner")
    parser.add_argument("--url",     default=DEFAULT_URL,    help="FastAPI base URL")
    parser.add_argument("--md-dir",  default=DEFAULT_MD_DIR, help="Directory with *.md files")
    parser.add_argument("--file",    default=None,            help="Run single .md file (e.g. C1.md)")
    parser.add_argument("--delay",   type=float, default=3.0, help="Seconds between turns")
    parser.add_argument("--verbose", action="store_true",     help="Print full agent replies + recs")
    args = parser.parse_args()

    # Health check
    print(f"{BOLD}Checking server at {args.url} ...{RESET}")
    try:
        r = requests.get(f"{args.url}/health", timeout=10)
        assert r.json().get("status") == "ok", f"Bad health response: {r.text}"
        print(f"{GREEN}Server is healthy.{RESET}\n")
    except Exception as e:
        print(f"{RED}Server not reachable: {e}{RESET}")
        sys.exit(1)

    # Collect files
    md_dir = Path(args.md_dir)
    if args.file:
        files = [md_dir / args.file]
    else:
        files = sorted(md_dir.glob("*.md"))

    if not files:
        print(f"{RED}No markdown files found in {md_dir}{RESET}")
        sys.exit(1)

    print(f"Found {len(files)} markdown file(s) to test.\n")

    # Run
    all_results = {}
    for md_file in files:
        try:
            conv = parse_markdown(md_file)
        except Exception as e:
            print(f"{RED}Failed to parse {md_file.name}: {e}{RESET}")
            continue
        results = run_conversation(conv, args.url, verbose=args.verbose, delay=args.delay)
        all_results[conv.file] = results

    print_summary(all_results)


if __name__ == "__main__":
    main()

