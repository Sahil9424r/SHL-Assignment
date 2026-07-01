"""
test_local.py
Quick test script to manually check your agent against sample conversations.
Run with: python test_local.py

Simulates the C1.md conversation and prints results.
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def send_chat(messages: list[dict]) -> dict:
    resp = requests.post(f"{BASE_URL}/chat", json={"messages": messages}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_conversation(turns: list[str], name: str = "Test"):
    """Simulate a multi-turn conversation by sending accumulated history."""
    print(f"\n{'='*60}")
    print(f"CONVERSATION: {name}")
    print('='*60)

    history = []

    for user_msg in turns:
        history.append({"role": "user", "content": user_msg})
        print(f"\nUSER: {user_msg}")

        result = send_chat(history)

        print(f"AGENT: {result['reply']}")
        if result["recommendations"]:
            print(f"RECOMMENDATIONS ({len(result['recommendations'])}):")
            for r in result["recommendations"]:
                print(f"  - [{r['test_type']}] {r['name']}")
                print(f"    {r['url']}")
        print(f"end_of_conversation: {result['end_of_conversation']}")

        # Add assistant reply to history
        history.append({"role": "assistant", "content": result["reply"]})

        if result["end_of_conversation"]:
            break


def main():
    # Check health first
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    assert resp.json() == {"status": "ok"}, "Health check failed!"
    print("Health check: OK")
    import time
    # --- C1: Senior leadership selection (from the markdown you shared) ---
    run_conversation(
        name="C1: Senior Leadership",
        turns=[
            "We need a solution for senior leadership.",
            "The pool consists of CXOs, director-level positions; people with more than 15 years of experience.",
            "Selection — comparing candidates against a leadership benchmark.",
            "Perfect, that's what we need.",
        ],
    )
    time.sleep(30)
    

    # --- Test: Vague query (should clarify, not recommend) ---
    run_conversation(
        name="Vague Query Test",
        turns=["I need an assessment."],
    )
    time.sleep(30)
    # --- Test: Java developer ---
    run_conversation(
        name="Java Developer",
        turns=[
            "Hiring a Java developer who works with stakeholders.",
            "Mid-level, around 4 years experience.",
        ],
    )
    time.sleep(30)
    # --- Test: Refinement ---
    run_conversation(
        name="Refinement Test",
        turns=[
            "I need cognitive tests for a data analyst role.",
            "Mid-level.",
            "Actually, also add personality tests.",
        ],
    )
    time.sleep(30)

    # --- Test: Off-topic refusal ---
    run_conversation(
        name="Off-topic Refusal",
        turns=["What is the best salary for a software engineer in India?"],
    )
    time.sleep(30)

def send_chat(messages: list[dict]) -> dict:
    resp = requests.post(f"{BASE_URL}/chat", json={"messages": messages}, timeout=30)
    if resp.status_code != 200:
        print(f"ERROR {resp.status_code}: {resp.text}")  # add this line
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    main()
