"""패턴 생성기(보너스).

크기 N(홀수)을 주면 N×N 십자가(Cross)와 X 패턴을 만든다.
  - Cross: 가운데 행과 가운데 열이 1
  - X    : 두 대각선이 1
성능 분석에서 크기별 입력을 만들 때 재활용한다.
"""

from matrix import Matrix, MatrixError


def _check_size(n):
    if not isinstance(n, int) or isinstance(n, bool) or n < 3 or n % 2 == 0:
        raise MatrixError(f"패턴 크기는 3 이상의 홀수여야 합니다(가운데 칸이 있어야 함): {n!r}")


def make_cross(n):
    _check_size(n)
    m = Matrix(n)
    mid = n // 2
    for i in range(n):
        m.set(mid, i, 1)
        m.set(i, mid, 1)
    return m


def make_x(n):
    _check_size(n)
    m = Matrix(n)
    for i in range(n):
        m.set(i, i, 1)
        m.set(i, n - 1 - i, 1)
    return m
