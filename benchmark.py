"""MAC 연산 시간 측정.

측정 구간은 '연산 함수 호출'만이다. 입력·출력·파일 읽기는 측정 밖에서 끝낸다.
time.perf_counter()는 시스템 시계 변경의 영향을 받지 않고
해상도가 가장 높은 표준 라이브러리 타이머라서 이것을 쓴다.
"""

import time
import unicodedata

from mac import mac_2d, mac_flat
from pattern_gen import make_cross, make_x

REPEAT = 10  # 미션 최소 기준: 크기별 10회 반복 측정 후 평균
SIZES = (3, 5, 13, 25)


def measure_ms(func, *args, repeat=REPEAT):
    """func(*args)를 repeat번 따로 재서 평균 시간(ms)을 돌려준다.

    첫 호출은 캐시·인터프리터 준비 영향이 있어 한 번 미리 돌리고(warm-up) 버린다.
    """
    func(*args)
    total = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        total += time.perf_counter() - start
    return total / repeat * 1000


def benchmark_sizes(filters=None, sizes=SIZES, repeat=REPEAT):
    """크기별 2D·1D MAC 평균 시간 표 데이터를 만든다.

    입력 패턴은 생성기로 만든 X 패턴(같은 크기면 매번 같은 입력)이고,
    필터는 data.json에서 읽은 Cross 필터가 있으면 그것을, 없으면 생성기 필터를 쓴다.
    1D는 펼치는 시간을 빼고 이미 펼친 배열로 MAC만 잰다(같은 입력, 같은 반복 횟수).
    """
    rows = []
    for n in sizes:
        pattern = make_x(n)
        if filters and n in filters:
            filt = filters[n]["Cross"]
            source = "data.json"
        else:
            filt = make_cross(n)
            source = "생성기"
        p2, f2 = pattern._rows, filt._rows
        p1, f1 = pattern.flatten(), filt.flatten()
        ms_2d = measure_ms(mac_2d, p2, f2, repeat=repeat)
        ms_1d = measure_ms(mac_flat, p1, f1, repeat=repeat)
        rows.append(
            {"n": n, "ops": n * n, "ms_2d": ms_2d, "ms_1d": ms_1d, "source": source}
        )
    return rows


def _width(text):
    """콘솔에서 차지하는 칸 수. 한글처럼 넓은 문자는 2칸으로 센다."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text, width, right=True):
    space = " " * max(width - _width(text), 0)
    return space + text if right else text + space


def format_table(rows):
    headers = ("크기", "평균(ms) 2D", "평균(ms) 1D", "연산 횟수(N²)", "2D ns/연산")
    widths = (8, 14, 14, 15, 12)
    lines = ["".join(_pad(h, w, i > 0) for i, (h, w) in enumerate(zip(headers, widths)))]
    lines.append("-" * sum(widths))
    for row in rows:
        per_op_ns = row["ms_2d"] * 1e6 / row["ops"]
        cells = (
            f"{row['n']}×{row['n']}",
            f"{row['ms_2d']:.4f}",
            f"{row['ms_1d']:.4f}",
            str(row["ops"]),
            f"{per_op_ns:.1f}",
        )
        lines.append("".join(_pad(c, w, i > 0) for i, (c, w) in enumerate(zip(cells, widths))))
    return "\n".join(lines)
