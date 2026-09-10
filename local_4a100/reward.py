# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from decimal import Decimal, InvalidOperation

_SOLUTION_CLIP_CHARS = 300
GSM8K_SCORER_VERSION = "final_numeric_v2"
_ANSWER_MARKER = re.compile(r"####(?!#)|\\boxed\b")
_NUMBER = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _numeric_token(value):
    """Accept a scalar number, not units, expressions, or arbitrary answer text."""
    token = str(value).strip()
    if token.startswith("$") and token.endswith("$"):
        token = token[1:-1].strip()
    if len(token) > 256 or _NUMBER.fullmatch(token) is None:
        return None
    return token.replace(",", "")


def _extract_strict(solution_str):
    # Only the final-response portion counts for models with explicit thinking tags.
    # A correct intermediate answer in <think> must not earn reward when the final
    # response is missing, truncated, or wrong.
    final_response = solution_str.rsplit("</think>", 1)[-1]
    if "<think>" in final_response:
        return None

    # Search the whole final response: the answer may be followed by >300 chars.
    # Do not fall back to an earlier correct answer if the last marker is invalid.
    marker = None
    for match in _ANSWER_MARKER.finditer(final_response):
        marker = match
    if marker is None:
        return None
    tail = final_response[marker.end():].lstrip(" \t")
    if marker.group().startswith("####"):
        return _numeric_token(tail.splitlines()[0] if tail else "")

    tail = tail.lstrip()
    if not tail.startswith("{"):
        return None
    depth = 0
    for index, char in enumerate(tail):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return _numeric_token(tail[1:index])
    return None


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        return _extract_strict(solution_str)

    # Preserve the legacy flexible mode; training uses strict explicit markers.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
    final_answer = None
    for candidate in reversed(answer):
        if candidate not in ["", "."]:
            final_answer = candidate
            break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: 'strict' accepts a final boxed scalar or #### numeric answer;
            'flexible' retains the legacy last-number heuristic.
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    answer = _numeric_token(answer) if answer is not None else None
    expected = _numeric_token(ground_truth)
    if answer is None or expected is None:
        return 0
    try:
        # Exact decimal equality, not an approximate float tolerance or eval().
        # This treats 10, 10.0 and 1e1 as the same numerical answer.
        correct = Decimal(answer) == Decimal(expected)
    except InvalidOperation:
        return 0
    return score if correct else format_score

def compute_dataset_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Use the corrected GSM8K scorer; leave MATH/DAPO and other routes unchanged."""
    if data_source == "openai/gsm8k":
        return compute_score(solution_str, ground_truth)
    from verl.utils.reward_score import default_compute_score
    return default_compute_score(
        data_source=data_source, solution_str=solution_str,
        ground_truth=ground_truth, extra_info=extra_info, **kwargs
    )
