"""n×n 정사각 2차원 배열(Matrix).

패턴과 필터를 모두 이 클래스 하나로 저장한다.
외부 라이브러리 없이 '리스트의 리스트'로 값을 담고,
특정 위치의 값을 읽고(get) 쓰는(set) 메서드를 제공한다.
"""

import math


class MatrixError(ValueError):
    """행렬 모양(행 수, 열 수)이나 값이 잘못됐을 때 쓰는 예외."""


class Matrix:
    """n×n 정사각 행렬. 값은 모두 float로 저장한다."""

    def __init__(self, size, fill=0.0):
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise MatrixError(f"크기는 1 이상의 정수여야 합니다: {size!r}")
        self.size = size
        # [[0.0]*n]*n 으로 만들면 모든 행이 같은 리스트를 가리킨다.
        # 그래서 행마다 새 리스트를 만든다.
        self._rows = [[float(fill)] * size for _ in range(size)]

    @classmethod
    def from_rows(cls, rows):
        """2차원 리스트를 검증해 Matrix로 만든다.

        - 리스트의 리스트인지
        - 정사각형인지(모든 행의 길이 == 행 수)
        - 모든 값이 유한한 숫자인지(bool, 문자열, NaN, inf 거부)
        중 하나라도 어기면 MatrixError를 낸다.
        """
        if not isinstance(rows, list) or not rows:
            raise MatrixError("2차원 배열(리스트의 리스트)이 아니거나 비어 있습니다.")
        size = len(rows)
        matrix = cls(size)
        for r, row in enumerate(rows):
            if not isinstance(row, list):
                raise MatrixError(f"{r + 1}행이 리스트가 아닙니다.")
            if len(row) != size:
                raise MatrixError(
                    f"정사각형이 아닙니다: 행 {size}개인데 {r + 1}행의 열이 {len(row)}개입니다."
                )
            for c, value in enumerate(row):
                matrix.set(r, c, value)
        return matrix

    def _check_index(self, r, c):
        if not (0 <= r < self.size and 0 <= c < self.size):
            raise IndexError(f"위치 ({r}, {c})는 {self.size}×{self.size} 범위를 벗어납니다.")

    def get(self, r, c):
        """r행 c열(0부터 셈)의 값을 읽는다."""
        self._check_index(r, c)
        return self._rows[r][c]

    def set(self, r, c, value):
        """r행 c열(0부터 셈)에 값을 저장한다. 숫자가 아니면 MatrixError."""
        self._check_index(r, c)
        # bool은 int의 하위 타입이라 True가 1로 들어가는 것을 막는다.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MatrixError(f"({r + 1}, {c + 1}) 값이 숫자가 아닙니다: {value!r}")
        if not math.isfinite(value):
            raise MatrixError(f"({r + 1}, {c + 1}) 값이 유한한 숫자가 아닙니다: {value!r}")
        self._rows[r][c] = float(value)

    def rows(self):
        """행 리스트의 복사본. 바깥에서 고쳐도 원본은 바뀌지 않는다."""
        return [row[:] for row in self._rows]

    def flatten(self):
        """행 우선(row-major) 순서로 펼친 길이 N²의 1차원 리스트."""
        flat = []
        for row in self._rows:
            flat.extend(row)
        return flat

    def __eq__(self, other):
        return isinstance(other, Matrix) and self._rows == other._rows

    def __repr__(self):
        return f"Matrix(size={self.size})"

    def to_text(self):
        """콘솔 출력용 문자열. 정수 값은 소수점 없이 보여 준다."""
        lines = []
        for row in self._rows:
            cells = [str(int(v)) if v.is_integer() else str(v) for v in row]
            lines.append(" ".join(cells))
        return "\n".join(lines)
