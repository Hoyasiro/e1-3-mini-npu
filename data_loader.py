"""data.json 로드, 스키마 검증, 케이스별 판정.

기대하는 구조
{
  "filters":  { "size_5":  { "cross": [[...]], "x": [[...]] }, ... },
  "patterns": { "size_5_1": { "input": [[...]], "expected": "+" }, ... }
}

원칙: 한 케이스가 잘못돼도 프로그램은 멈추지 않는다.
잘못된 필터는 '로드 실패'로 기록하고 건너뛰며,
잘못된 패턴은 그 케이스만 FAIL로 처리하고 사유를 남긴다.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

from labels import CROSS, X, normalize_label
from mac import decide, mac
from matrix import Matrix, MatrixError

FILTER_KEY = re.compile(r"^size_(\d+)$")
PATTERN_KEY = re.compile(r"^size_(\d+)_(\d+)$")


class DataError(Exception):
    """파일 전체를 읽을 수 없을 때(파일 없음, JSON 문법 오류, 최상위 구조 오류)."""


@dataclass
class FilterLoad:
    key: str
    size: Optional[int]
    ok: bool
    message: str


@dataclass
class CaseResult:
    case_id: str
    cross_score: Optional[float] = None
    x_score: Optional[float] = None
    decision: Optional[str] = None  # Cross / X / UNDECIDED, 계산 못 했으면 None
    expected: Optional[str] = None  # 표준 라벨, 해석 못 했으면 None
    passed: bool = False
    reason: str = ""


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise DataError(f"파일을 찾을 수 없습니다: {path}")
    except json.JSONDecodeError as e:
        raise DataError(f"JSON 문법 오류: {e.msg} ({e.lineno}행 {e.colno}열)")
    except OSError as e:
        raise DataError(f"파일을 읽을 수 없습니다: {e}")
    if not isinstance(data, dict):
        raise DataError("최상위 구조가 객체({})가 아닙니다.")
    for section in ("filters", "patterns"):
        if not isinstance(data.get(section), dict):
            raise DataError(f"'{section}' 항목이 없거나 객체({{}})가 아닙니다.")
    return data


def load_filters(data):
    """필터를 읽어 {N: {'Cross': Matrix, 'X': Matrix}} 와 로드 기록 목록을 돌려준다."""
    filters = {}
    logs = []
    for key, value in data["filters"].items():
        m = FILTER_KEY.match(key)
        if not m:
            logs.append(FilterLoad(key, None, False, "키 형식 오류(size_N 아님)"))
            continue
        n = int(m.group(1))
        if not isinstance(value, dict):
            logs.append(FilterLoad(key, n, False, "필터 값이 객체({})가 아님"))
            continue
        pair = {}
        error = None
        for raw_label, rows in value.items():
            label = normalize_label(raw_label)
            if label is None:
                error = f"필터 라벨 해석 불가: {raw_label!r}"
                break
            if label in pair:
                error = f"필터 라벨 중복: {raw_label!r}"
                break
            try:
                matrix = Matrix.from_rows(rows)
            except MatrixError as e:
                error = f"{label} 필터 형식 오류: {e}"
                break
            if matrix.size != n:
                error = f"{label} 필터 크기 불일치: 키 N={n}, 실제 {matrix.size}×{matrix.size}"
                break
            pair[label] = matrix
        if error is None and set(pair) != {CROSS, X}:
            missing = [lb for lb in (CROSS, X) if lb not in pair]
            error = f"필터 누락: {', '.join(missing)}"
        if error:
            logs.append(FilterLoad(key, n, False, error))
            continue
        filters[n] = pair
        logs.append(FilterLoad(key, n, True, "필터 로드 완료 (Cross, X)"))
    return filters, logs


def analyze_case(case_id, entry, filters):
    """패턴 한 건을 검증하고 판정한다. 어떤 오류도 예외로 새지 않고 CaseResult로 끝난다."""
    result = CaseResult(case_id)

    m = PATTERN_KEY.match(case_id)
    if not m:
        result.reason = "키 형식 오류: size_{N}_{idx} 형태가 아님"
        return result
    n = int(m.group(1))

    if not isinstance(entry, dict):
        result.reason = "스키마 오류: 패턴 항목이 객체({})가 아님"
        return result

    result.expected = normalize_label(entry.get("expected"))
    if "expected" not in entry:
        result.reason = "스키마 오류: expected 없음"
        return result
    if result.expected is None:
        result.reason = f"라벨 오류: expected {entry.get('expected')!r}를 Cross/X로 해석할 수 없음"
        return result

    if n not in filters:
        result.reason = f"필터 없음: size_{n} 필터가 로드되지 않음"
        return result

    if "input" not in entry:
        result.reason = "스키마 오류: input 없음"
        return result
    try:
        pattern = Matrix.from_rows(entry["input"])
    except MatrixError as e:
        result.reason = f"입력 형식 오류: {e}"
        return result
    if pattern.size != n:
        result.reason = (
            f"크기 불일치: 키 N={n}인데 패턴은 {pattern.size}×{pattern.size} "
            f"(size_{n} 필터와 계산 불가)"
        )
        return result

    pair = filters[n]
    result.cross_score = mac(pattern, pair[CROSS])
    result.x_score = mac(pattern, pair[X])
    winner = decide(result.cross_score, result.x_score)
    result.decision = {"A": CROSS, "B": X, None: "UNDECIDED"}[winner]

    if result.decision == "UNDECIDED":
        result.reason = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
    elif result.decision == result.expected:
        result.passed = True
    else:
        result.reason = f"판정 {result.decision} ≠ expected {result.expected}"
    return result


def analyze_patterns(data, filters):
    return [analyze_case(key, entry, filters) for key, entry in data["patterns"].items()]
