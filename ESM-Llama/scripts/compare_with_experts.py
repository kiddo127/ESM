#!/usr/bin/env python3
"""
Compare merged model evaluation results with expert model baselines.
Calculate percentage of expert performance retained by merged model.
"""
import json
import os
import sys
from pathlib import Path


# Expert model result directories
EXPERT_DIRS = {
    'math': 'results/llama-3.2-3b_math',
    'instruction': 'results/llama-3.2-3b_instruction',
    'coding': 'results/llama-3.2-3b_coding',
    'multilingual': 'results/llama-3.2-3b_multilingual',
    'safety': 'results/llama-3.2-3b_safety',
}

# Metrics to extract for each test
# For tasks with multiple metrics, average them


def find_json_file(directory, pattern='*.json'):
    """Find JSON file in directory or subdirectories."""
    path = Path(directory)
    if not path.exists():
        return None
    # First check direct JSON files
    for f in path.glob(pattern):
        return f
    # Then check subdirectories
    for f in path.rglob(pattern):
        return f
    return None


def load_json(filepath):
    """Load JSON file."""
    if filepath is None:
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def get_gsm8k_score(data):
    """Extract gsm8k_cot flexible-extract score."""
    if data is None:
        return None
    results = data.get('results', {})
    gsm8k = results.get('gsm8k_cot', {})
    return gsm8k.get('exact_match,flexible-extract')


def get_ifeval_score(data):
    """Extract ifeval scores and average them."""
    if data is None:
        return None
    results = data.get('results', {})
    ifeval = results.get('ifeval', {})
    prompt_loose = ifeval.get('prompt_level_loose_acc,none')
    inst_loose = ifeval.get('inst_level_loose_acc,none')
    if prompt_loose is None or inst_loose is None:
        return None
    return (prompt_loose + inst_loose) / 2


def get_bigcode_score(data):
    """Extract humanevalplus and mbppplus pass@1 and average."""
    if data is None:
        return None
    humaneval = data.get('humanevalplus', {})
    mbpp = data.get('mbppplus', {})
    humaneval_p1 = humaneval.get('pass@1')
    mbpp_p1 = mbpp.get('pass@1')
    if humaneval_p1 is None or mbpp_p1 is None:
        return None
    return (humaneval_p1 + mbpp_p1) / 2


def get_multilingual_score(data):
    """Extract all 12 multilingual task scores and average."""
    if data is None:
        return None
    results = data.get('results', {})
    # 12 tasks: m_mmlu_fr, arc_fr, hellaswag_fr, m_mmlu_es, arc_es, hellaswag_es,
    #           m_mmlu_de, arc_de, hellaswag_de, m_mmlu_ru, arc_ru, hellaswag_ru
    tasks = ['m_mmlu_fr', 'arc_fr', 'hellaswag_fr',
             'm_mmlu_es', 'arc_es', 'hellaswag_es',
             'm_mmlu_de', 'arc_de', 'hellaswag_de',
             'm_mmlu_ru', 'arc_ru', 'hellaswag_ru']
    scores = []
    for task in tasks:
        task_data = results.get(task, {})
        acc = task_data.get('acc,none')
        if acc is not None:
            scores.append(acc)
    if len(scores) == 0:
        return None
    return sum(scores) / len(scores)


def get_safety_score(data):
    """Extract safety scores and average."""
    if data is None:
        return None
    wildguardtest = data.get('wildguardtest', {})
    harmbench = data.get('harmbench', {})
    xstest = data.get('xstest', {})
    do_anything_now = data.get('do_anything_now', {})

    # 1 - micro harm (lower) for wildguardtest
    micro_harm = wildguardtest.get('micro harm (lower)')
    score1 = 1 - micro_harm if micro_harm is not None else None

    # 1 - micro ASR (lower) for harmbench
    micro_asr = harmbench.get('micro ASR (lower)')
    score2 = 1 - micro_asr if micro_asr is not None else None

    # overall_accuracy for xstest
    score3 = xstest.get('overall_accuracy')

    # 1 - macro ASR for do_anything_now
    macro_asr = do_anything_now.get('macro ASR')
    score4 = 1 - macro_asr if macro_asr is not None else None

    scores = [s for s in [score1, score2, score3, score4] if s is not None]
    if len(scores) == 0:
        return None
    return sum(scores) / len(scores)


def get_expert_scores(base_dir):
    """Load expert model scores from results directory."""
    scores = {}

    # Math expert -> gsm8k
    math_dir = os.path.join(base_dir, EXPERT_DIRS['math'])
    math_json = find_json_file(math_dir)
    math_data = load_json(math_json)
    scores['gsm8k_cot'] = get_gsm8k_score(math_data)

    # Instruction expert -> ifeval
    inst_dir = os.path.join(base_dir, EXPERT_DIRS['instruction'])
    inst_json = find_json_file(inst_dir)
    inst_data = load_json(inst_json)
    scores['ifeval'] = get_ifeval_score(inst_data)

    # Coding expert -> bigcode
    coding_dir = os.path.join(base_dir, EXPERT_DIRS['coding'])
    coding_json = find_json_file(coding_dir, 'code_eval.json')
    coding_data = load_json(coding_json)
    scores['bigcode'] = get_bigcode_score(coding_data)

    # Multilingual expert -> multilingual
    multi_dir = os.path.join(base_dir, EXPERT_DIRS['multilingual'])
    multi_json = find_json_file(multi_dir)
    multi_data = load_json(multi_json)
    scores['multilingual'] = get_multilingual_score(multi_data)

    # Safety expert -> safety
    safety_dir = os.path.join(base_dir, EXPERT_DIRS['safety'])
    safety_json = find_json_file(safety_dir, 'safety_eval.json')
    safety_data = load_json(safety_json)
    scores['safety'] = get_safety_score(safety_data)

    return scores


def get_merged_scores(output_path, base_dir):
    """Load merged model scores from evaluation output."""
    scores = {}

    output_dir = os.path.join(base_dir, output_path)

    # Find lm_eval results
    lm_eval_jsons = list(Path(output_dir).rglob('results_*.json'))
    for j in lm_eval_jsons:
        data = load_json(j)
        if data:
            results = data.get('results', {})
            if 'gsm8k_cot' in results:
                scores['gsm8k_cot'] = get_gsm8k_score(data)
            if 'ifeval' in results:
                scores['ifeval'] = get_ifeval_score(data)
            # Check for multilingual
            if 'm_mmlu_fr' in results:
                scores['multilingual'] = get_multilingual_score(data)

    # Find code eval results
    code_json = Path(output_dir) / 'code_eval.json'
    if code_json.exists():
        code_data = load_json(code_json)
        scores['bigcode'] = get_bigcode_score(code_data)

    # Find safety eval results
    safety_json = Path(output_dir) / 'safety_eval.json'
    if safety_json.exists():
        safety_data = load_json(safety_json)
        scores['safety'] = get_safety_score(safety_data)

    return scores


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_with_experts.py <merged_output_path>")
        sys.exit(1)

    output_path = sys.argv[1]
    # Use project root as base directory
    base_dir = os.environ.get('PROJECT_ROOT', str(Path(__file__).resolve().parents[1]))

    expert_scores = get_expert_scores(base_dir)
    merged_scores = get_merged_scores(output_path, base_dir)

    print("\n========================================")
    print("Performance Comparison (Merged vs Expert)")
    print("========================================")

    tasks = ['gsm8k_cot', 'ifeval', 'bigcode', 'multilingual', 'safety']
    task_names = ['Math (gsm8k)', 'Instruction (ifeval)', 'Coding (bigcode)',
                  'Multilingual', 'Safety']

    for task, name in zip(tasks, task_names):
        expert = expert_scores.get(task)
        merged = merged_scores.get(task)

        if expert is None:
            print(f"[{name}] Expert baseline not found")
            continue
        if merged is None:
            print(f"[{name}] Merged result not found")
            continue

        ratio = (merged / expert) * 100 if expert != 0 else 0
        print(f"[{name}] Expert: {expert*100:.2f}%, Merged: {merged*100:.2f}%, Ratio: {ratio:.1f}%")

    print("========================================\n")


if __name__ == "__main__":
    main()