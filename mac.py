"""MAC(Multiply-Accumulate) 연산과 점수 비교 정책.

- mac_2d   : 2차원 리스트를 이중 반복문으로 곱하고 더한다(기본 구현).
- mac_flat : 1차원(길이 N²) 리스트를 단일 반복문으로 곱하고 더한다(보너스 최적화).
- mac      : Matrix 두 개를 받아 크기를 확인한 뒤 mac_2d를 부른다.
- decide   : 두 점수를 epsilon 허용오차로 비교해 'A' / 'B' / None(동점)을 돌려준다.

NumPy 같은 벡터화 라이브러리 없이 for 문으로만 계산한다.
"""

EPSILON = 1e-9


def mac_2d(pattern_rows, filter_rows):
    """같은 위치끼리 곱해서 모두 더한다. 연산 횟수는 곱셈 N², 덧셈 N²."""
    total = 0.0
    n = len(pattern_rows)
    for r in range(n):
        p_row = pattern_rows[r]
        f_row = filter_rows[r]
        for c in range(n):
            total += p_row[c] * f_row[c]
    return total


def mac_flat(pattern_flat, filter_flat):
    """1차원으로 펼친 두 배열의 MAC. 인덱스 한 번으로 값에 접근한다."""
    total = 0.0
    for i in range(len(pattern_flat)):
        total += pattern_flat[i] * filter_flat[i]
    return total


def mac(pattern, filt):
    """Matrix 두 개의 MAC 점수. 크기가 다르면 ValueError."""
    if pattern.size != filt.size:
        raise ValueError(
            f"크기 불일치: 패턴 {pattern.size}×{pattern.size}, 필터 {filt.size}×{filt.size}"
        )
    return mac_2d(pattern._rows, filt._rows)


def decide(score_a, score_b, epsilon=EPSILON):
    """점수가 큰 쪽을 고른다.

    |A - B| < epsilon 이면 동점으로 보고 None을 돌려준다.
    부동소수점 계산 오차(예: 0.9 와 0.8999999999999999)로
    한쪽이 '아주 조금' 큰 것을 승리로 오판하지 않기 위해서다.
    """
    if abs(score_a - score_b) < epsilon:
        return None
    return "A" if score_a > score_b else "B"
