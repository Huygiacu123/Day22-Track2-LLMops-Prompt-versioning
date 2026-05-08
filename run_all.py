"""
Run all steps of the Day 22 Lab sequentially.

Usage:
  python run_all.py              # Run all steps
  python run_all.py --step 1     # Run only step 1
  python run_all.py --step 3     # Run only step 3
"""

import sys
import subprocess
import time

STEPS = {
    1: ("01_langsmith_rag_pipeline.py", "LangSmith RAG Pipeline"),
    2: ("02_prompt_hub_ab_routing.py", "Prompt Hub A/B Routing"),
    3: ("03_ragas_evaluation.py", "RAGAS Evaluation"),
    4: ("04_guardrails_validator.py", "Guardrails AI Validators"),
}


def run_step(step_num: int):
    """Run a single step."""
    if step_num not in STEPS:
        print(f"❌ Invalid step: {step_num}. Valid steps: 1-4")
        return False
    
    script, description = STEPS[step_num]
    
    print(f"\n{'='*70}")
    print(f"  Running Step {step_num}: {description}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            check=True,
            cwd="."
        )
        print(f"\n✅ Step {step_num} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step {step_num} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Step {step_num} failed: {e}")
        return False


def main():
    """Main entry point."""
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--step":
        if len(sys.argv) < 3:
            print("Usage: python run_all.py --step <1-4>")
            sys.exit(1)
        
        try:
            step_num = int(sys.argv[2])
            run_step(step_num)
        except ValueError:
            print(f"Invalid step number: {sys.argv[2]}")
            sys.exit(1)
    else:
        # Run all steps
        print("=" * 70)
        print("  Day 22 Lab: LangSmith + Prompt Versioning + RAGAS + Guardrails")
        print("=" * 70)
        
        results = {}
        for step_num in sorted(STEPS.keys()):
            results[step_num] = run_step(step_num)
            time.sleep(1)  # Brief pause between steps
        
        # Summary
        print("\n" + "=" * 70)
        print("  Summary")
        print("=" * 70)
        
        for step_num in sorted(STEPS.keys()):
            status = "✅ PASS" if results[step_num] else "❌ FAIL"
            _, description = STEPS[step_num]
            print(f"  Step {step_num}: {description:40s} {status}")
        
        all_passed = all(results.values())
        print("\n" + "=" * 70)
        if all_passed:
            print("✅ All steps completed successfully!")
        else:
            print("⚠️  Some steps failed. Check the output above.")
        print("=" * 70)


if __name__ == "__main__":
    main()
