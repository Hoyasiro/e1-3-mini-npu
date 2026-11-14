"""Mini NPU Simulator - 콘솔 진입점.

실행: python main.py            (data.json은 main.py와 같은 폴더에서 찾는다)
      python main.py --data 경로  (다른 data.json 지정)
"""

import argparse
import sys
from pathlib import Path

import benchmark
from benchmark import benchmark_sizes, format_table, measure_ms
from data_loader import DataError, analyze_patterns, load_filters, load_json
from labels import CROSS, X
from mac import EPSILON, decide, mac, mac_2d
from matrix import Matrix
from pattern_gen import make_cross, make_x

DEFAULT_DATA = Path(__file__).resolve().parent / "data.json"
LINE = "#" + "-" * 40


def section(title):
    print()
    print(LINE)
    print(f"# {title}")
    print(LINE)


# ---------------------------------------------------------------- 입력 도우미

def parse_row(text, n):
    """공백으로 구분된 숫자 n개를 float 리스트로 바꾼다. 형식이 틀리면 ValueError."""
    message = f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요."
    parts = text.split()
    if len(parts) != n:
        raise ValueError(f"{message} (입력한 개수: {len(parts)})")
    values = []
    for part in parts:
        try:
            value = float(part)
        except ValueError:
            raise ValueError(f"{message} (숫자가 아님: {part!r})")
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{message} (유한한 숫자가 아님: {part!r})")
        values.append(value)
    return values


def read_matrix(title, n=3):
    """n줄을 한 줄씩 입력받아 Matrix로 만든다. 틀린 줄만 다시 입력받는다."""
    print(f"{title} ({n}줄 입력, 공백 구분)")
    rows = []
    while len(rows) < n:
        text = input(f"  {len(rows) + 1}행> ")
        try:
            rows.append(parse_row(text, n))
        except ValueError as e:
            print(f"  {e}")
            print(f"  {len(rows) + 1}행을 다시 입력하세요.")
    return Matrix.from_rows(rows)


# ---------------------------------------------------------------- 모드 1

def run_user_mode():
    section("[1] 필터 입력")
    filter_a = read_matrix("필터 A")
    print()
    filter_b = read_matrix("필터 B")
    print("\n저장 확인")
    print("필터 A:")
    print(filter_a.to_text())
    print("필터 B:")
    print(filter_b.to_text())

    section("[2] 패턴 입력")
    pattern = read_matrix("패턴")

    section("[3] MAC 결과")
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    print(f"A 점수: {score_a!r}")
    print(f"B 점수: {score_b!r}")
    # 패턴 1개를 필터 A, B에 각각 대는 한 번의 판정을 10회 재서 평균을 낸다.
    p, a, b = pattern._rows, filter_a._rows, filter_b._rows
    elapsed = measure_ms(lambda: (mac_2d(p, a), mac_2d(p, b)), repeat=benchmark.REPEAT)
    print(f"연산 시간(평균/{benchmark.REPEAT}회): {elapsed:.4f} ms")
    winner = decide(score_a, score_b)
    if winner is None:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON:g})".replace("e-0", "e-"))
    else:
        print(f"판정: {winner}")

    section(f"[4] 성능 분석 (3×3, 평균/{benchmark.REPEAT}회)")
    print(format_table(benchmark_sizes(sizes=(3,), repeat=benchmark.REPEAT)))


# ---------------------------------------------------------------- 모드 2

def run_json_mode(path):
    section("[1] 필터 로드")
    try:
        data = load_json(path)
    except DataError as e:
        print(f"✗ data.json을 읽을 수 없습니다: {e}")
        return
    filters, logs = load_filters(data)
    for log in logs:
        mark = "✓" if log.ok else "✗"
        print(f"{mark} {log.key:<8} {log.message}")

    section("[2] 패턴 분석 (라벨 정규화 적용)")
    results = analyze_patterns(data, filters)
    for r in results:
        print(f"--- {r.case_id} ---")
        if r.decision is None:
            # 점수를 계산하기 전에 검증에서 걸린 경우
            print(f"FAIL ({r.reason})")
            continue
        print(f"Cross 점수: {r.cross_score!r}")
        print(f"X 점수: {r.x_score!r}")
        status = "PASS" if r.passed else f"FAIL ({r.reason})"
        print(f"판정: {r.decision} | expected: {r.expected} | {status}")

    section(f"[3] 성능 분석 (평균/{benchmark.REPEAT}회)")
    print(format_table(benchmark_sizes(filters, repeat=benchmark.REPEAT)))
    print("2D: 2차원 리스트 이중 반복문, 1D: 길이 N² 배열 단일 반복문(보너스)")

    section("[4] 결과 요약")
    failed = [r for r in results if not r.passed]
    print(f"총 테스트: {len(results)}개")
    print(f"통과: {len(results) - len(failed)}개")
    print(f"실패: {len(failed)}개")
    if failed:
        print("\n실패 케이스:")
        for r in failed:
            print(f"- {r.case_id}: {r.reason}")
    print('\n(상세 원인 분석 및 복잡도 설명은 README.md의 "결과 리포트" 섹션에 작성)')


# ---------------------------------------------------------------- 모드 3 (보너스)

def run_generator_mode():
    section("[1] 패턴 생성기")
    while True:
        text = input("크기 N (3 이상의 홀수): ").strip()
        try:
            n = int(text)
            cross, x = make_cross(n), make_x(n)
            break
        except ValueError as e:
            print(f"입력 오류: {e if str(e).startswith('패턴') else '정수를 입력하세요.'}")
    if n <= 13:
        print(f"\nCross {n}×{n}:\n{cross.to_text()}")
        print(f"\nX {n}×{n}:\n{x.to_text()}")
    else:
        print("(13×13보다 크면 패턴 출력은 생략합니다)")

    section("[2] 생성 패턴으로 MAC 판정")
    for name, pattern in ((CROSS, cross), (X, x)):
        cs, xs = mac(pattern, cross), mac(pattern, x)
        winner = {"A": CROSS, "B": X, None: "UNDECIDED"}[decide(cs, xs)]
        print(f"{name} 패턴 → Cross 점수 {cs!r}, X 점수 {xs!r}, 판정 {winner}")

    section(f"[3] 성능 분석 ({n}×{n}, 평균/{benchmark.REPEAT}회)")
    print(format_table(benchmark_sizes(sizes=(n,), repeat=benchmark.REPEAT)))


# ---------------------------------------------------------------- 메뉴

def main(argv=None):
    parser = argparse.ArgumentParser(description="Mini NPU Simulator")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="data.json 경로")
    parser.add_argument(
        "--repeat", type=int, default=benchmark.REPEAT,
        help="성능 분석 반복 횟수(기본 10, 최소 10)",
    )
    args = parser.parse_args(argv)
    if args.repeat < 10:
        parser.error("--repeat는 10 이상이어야 합니다(미션 최소 기준).")
    benchmark.REPEAT = args.repeat

    # 출력이 파일·파이프로 갈 때 Windows 기본 인코딩(cp949)에 없는 문자(✓ 등)가
    # 나오면 UnicodeEncodeError로 멈춘다. 그런 문자는 '?'로 바꿔 계속 진행한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    print("=== Mini NPU Simulator ===")
    modes = {
        "1": run_user_mode,
        "2": lambda: run_json_mode(args.data),
        "3": run_generator_mode,
    }
    while True:
        print("\n[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        print("3. 패턴 생성기 (보너스)")
        print("0. 종료")
        choice = input("선택: ").strip()
        if choice == "0":
            print("종료합니다.")
            return
        if choice not in modes:
            print("0~3 중에서 고르세요.")
            continue
        modes[choice]()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        # Ctrl+C, Ctrl+Z(Windows)/Ctrl+D 로 입력이 끊겨도 traceback 없이 끝낸다.
        print("\n입력이 중단되어 종료합니다.")
        sys.exit(0)
