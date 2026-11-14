"""Mini NPU Simulator 단위 테스트 (표준 라이브러리 unittest).

실행: python -m unittest discover -s tests -v   (프로젝트 루트에서)
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import benchmark_sizes, measure_ms  # noqa: E402
from data_loader import (  # noqa: E402
    DataError, analyze_case, analyze_patterns, load_filters, load_json,
)
from labels import CROSS, X, normalize_label  # noqa: E402
from mac import EPSILON, decide, mac, mac_2d, mac_flat  # noqa: E402
from main import parse_row  # noqa: E402
from matrix import Matrix, MatrixError  # noqa: E402
from pattern_gen import make_cross, make_x  # noqa: E402

CROSS3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
X3 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


class MatrixTest(unittest.TestCase):
    def test_get_set(self):
        m = Matrix(3)
        m.set(2, 1, 7)
        self.assertEqual(m.get(2, 1), 7.0)
        self.assertEqual(m.get(0, 0), 0.0)

    def test_rows_are_independent(self):
        m = Matrix(3)
        m.set(0, 0, 1)
        self.assertEqual(m.get(1, 0), 0.0)  # [[0]*n]*n 공유 버그가 없어야 함

    def test_out_of_range(self):
        with self.assertRaises(IndexError):
            Matrix(3).get(3, 0)
        with self.assertRaises(IndexError):
            Matrix(3).set(-1, 0, 1)

    def test_from_rows_rejects_non_square(self):
        with self.assertRaises(MatrixError):
            Matrix.from_rows([[1, 0, 1], [0, 1]])
        with self.assertRaises(MatrixError):
            Matrix.from_rows([[1, 0], [0, 1], [1, 1]])

    def test_from_rows_rejects_bad_values(self):
        for bad in ("1", True, None, float("nan"), float("inf")):
            with self.assertRaises(MatrixError, msg=repr(bad)):
                Matrix.from_rows([[bad]])

    def test_supported_sizes(self):
        for n in (3, 5, 13, 25):
            self.assertEqual(Matrix(n).size, n)
            self.assertEqual(len(Matrix(n).flatten()), n * n)

    def test_rows_returns_copy(self):
        m = Matrix.from_rows(CROSS3)
        m.rows()[0][0] = 9
        self.assertEqual(m.get(0, 0), 0.0)


class MacTest(unittest.TestCase):
    def test_mission_examples(self):
        cross, x = Matrix.from_rows(CROSS3), Matrix.from_rows(X3)
        self.assertEqual(mac(cross, cross), 5.0)
        self.assertEqual(mac(cross, x), 1.0)

    def test_flat_equals_2d(self):
        for n in (3, 5, 13, 25):
            p, f = make_x(n), make_cross(n)
            self.assertEqual(mac_flat(p.flatten(), f.flatten()), mac_2d(p.rows(), f.rows()))

    def test_size_mismatch(self):
        with self.assertRaises(ValueError):
            mac(Matrix(3), Matrix(5))

    def test_float_scores(self):
        p = Matrix.from_rows([[0.5, 0.5], [0.5, 0.5]])
        f = Matrix.from_rows([[0.2, 0.2], [0.2, 0.2]])
        self.assertAlmostEqual(mac(p, f), 0.4)


class DecideTest(unittest.TestCase):
    def test_winner(self):
        self.assertEqual(decide(5.0, 1.0), "A")
        self.assertEqual(decide(1.0, 5.0), "B")

    def test_exact_tie(self):
        self.assertIsNone(decide(3.0, 3.0))

    def test_float_error_is_tie(self):
        # 0.1+0.2 == 0.30000000000000004 이라 == 비교로는 동점이 아니다.
        self.assertNotEqual(0.1 + 0.2, 0.3)
        self.assertIsNone(decide(0.1 + 0.2, 0.3))
        self.assertIsNone(decide(0.9, 0.8999999999999999))

    def test_epsilon_boundary(self):
        self.assertIsNone(decide(1.0, 1.0 + EPSILON / 2))
        self.assertEqual(decide(1.0 + EPSILON * 2, 1.0), "A")


class LabelTest(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_label("+"), CROSS)
        self.assertEqual(normalize_label("cross"), CROSS)
        self.assertEqual(normalize_label("Cross"), CROSS)
        self.assertEqual(normalize_label("x"), X)
        self.assertEqual(normalize_label(" X "), X)

    def test_unknown(self):
        for bad in ("o", "", "plus", None, 1):
            self.assertIsNone(normalize_label(bad), repr(bad))


class PatternGenTest(unittest.TestCase):
    def test_3x3_matches_mission(self):
        self.assertEqual(make_cross(3).rows(), CROSS3)
        self.assertEqual(make_x(3).rows(), X3)

    def test_scores_grow_with_n(self):
        for n in (5, 13, 25):
            self.assertEqual(mac(make_cross(n), make_cross(n)), 2 * n - 1)
            self.assertEqual(mac(make_cross(n), make_x(n)), 1.0)

    def test_rejects_even_or_small(self):
        for bad in (2, 4, 1, 0, -3):
            with self.assertRaises(MatrixError):
                make_cross(bad)


class ParseRowTest(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(parse_row(" 0 1  0 ", 3), [0.0, 1.0, 0.0])
        self.assertEqual(parse_row("0.5 -1 1e-3", 3), [0.5, -1.0, 0.001])

    def test_wrong_count(self):
        for text in ("1 1", "1 1 1 1", ""):
            with self.assertRaises(ValueError) as ctx:
                parse_row(text, 3)
            self.assertIn("입력 형식 오류", str(ctx.exception))

    def test_not_number(self):
        for text in ("a 1 0", "1,0,1", "nan 1 0", "inf 0 0"):
            with self.assertRaises(ValueError):
                parse_row(text, 3)


def _data(**patterns):
    return {
        "filters": {
            "size_5": {"cross": make_cross(5).rows(), "x": make_x(5).rows()},
        },
        "patterns": patterns,
    }


class DataLoaderTest(unittest.TestCase):
    def test_pass_cases(self):
        data = _data(
            size_5_1={"input": make_cross(5).rows(), "expected": "+"},
            size_5_2={"input": make_x(5).rows(), "expected": "x"},
        )
        filters, logs = load_filters(data)
        self.assertTrue(all(log.ok for log in logs))
        results = analyze_patterns(data, filters)
        self.assertEqual([r.decision for r in results], [CROSS, X])
        self.assertTrue(all(r.passed for r in results))

    def test_failures_do_not_raise(self):
        data = _data(
            size_5_1={"input": make_cross(3).rows(), "expected": "+"},  # 크기 불일치
            size_13_1={"input": make_cross(13).rows(), "expected": "+"},  # 필터 없음
            size_5_2={"input": make_cross(5).rows(), "expected": "o"},  # 라벨 오류
            size_5_3={"input": [[1, 0], [0]], "expected": "+"},  # 비정사각
            size_5_4={"expected": "+"},  # input 없음
            bad_key={"input": make_cross(5).rows(), "expected": "+"},  # 키 형식
            size_5_5="not a dict",  # 스키마 오류
        )
        filters, _ = load_filters(data)
        results = analyze_patterns(data, filters)
        self.assertEqual(len(results), 7)
        for r in results:
            self.assertFalse(r.passed, r.case_id)
            self.assertTrue(r.reason, r.case_id)
        self.assertIn("크기 불일치", results[0].reason)
        self.assertIn("필터 없음", results[1].reason)
        self.assertIn("라벨 오류", results[2].reason)

    def test_tie_is_fail(self):
        filters = {3: {CROSS: Matrix.from_rows(CROSS3), X: Matrix.from_rows(CROSS3)}}
        r = analyze_case("size_3_1", {"input": CROSS3, "expected": "+"}, filters)
        self.assertEqual(r.decision, "UNDECIDED")
        self.assertFalse(r.passed)

    def test_wrong_decision_is_fail(self):
        filters, _ = load_filters(_data())
        r = analyze_case("size_5_1", {"input": make_x(5).rows(), "expected": "+"}, filters)
        self.assertEqual(r.decision, X)
        self.assertFalse(r.passed)

    def test_bad_filters_are_skipped(self):
        data = {
            "filters": {
                "size_5": {"cross": make_cross(5).rows()},  # X 없음
                "size_13": {"cross": make_cross(5).rows(), "x": make_x(5).rows()},  # 크기
                "size_3": {"plus": CROSS3, "x": X3},  # 라벨
                "five": {},  # 키
            },
            "patterns": {},
        }
        filters, logs = load_filters(data)
        self.assertEqual(filters, {})
        self.assertEqual([log.ok for log in logs], [False] * 4)

    def test_load_json_errors(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "none.json")
            with self.assertRaises(DataError):
                load_json(missing)
            broken = os.path.join(d, "broken.json")
            with open(broken, "w", encoding="utf-8") as f:
                f.write("{ not json")
            with self.assertRaises(DataError):
                load_json(broken)
            no_filters = os.path.join(d, "nf.json")
            with open(no_filters, "w", encoding="utf-8") as f:
                json.dump({"patterns": {}}, f)
            with self.assertRaises(DataError):
                load_json(no_filters)


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data.json")


@unittest.skipUnless(os.path.exists(DATA_PATH), "data.json 없음")
class OfficialDataTest(unittest.TestCase):
    """미션 제공 data.json으로 README 결과 리포트의 숫자를 재현한다."""

    def setUp(self):
        data = load_json(DATA_PATH)
        self.filters, self.logs = load_filters(data)
        self.results = {r.case_id: r for r in analyze_patterns(data, self.filters)}

    def test_all_filters_loaded(self):
        self.assertEqual(sorted(self.filters), [5, 13, 25])
        self.assertTrue(all(log.ok for log in self.logs))

    def test_summary(self):
        passed = sorted(k for k, r in self.results.items() if r.passed)
        failed = sorted(k for k, r in self.results.items() if not r.passed)
        self.assertEqual(passed, ["size_13_1", "size_25_2", "size_5_2"])
        self.assertEqual(failed, ["size_13_2", "size_25_1", "size_5_1"])
        for k in failed:
            self.assertEqual(self.results[k].decision, "UNDECIDED")

    def test_ties_are_float_noise_only(self):
        # 실패 3건은 실제로는 같은 값이고 차이는 누적 오차(1e-14 미만)뿐이다.
        for k in ("size_5_1", "size_13_2", "size_25_1"):
            r = self.results[k]
            self.assertLess(abs(r.cross_score - r.x_score), 1e-14)

    def test_plain_comparison_would_mislabel(self):
        # epsilon 없이 > 로만 비교하면 3건 모두 expected와 다른 라벨을 고른다.
        for k in ("size_5_1", "size_13_2", "size_25_1"):
            r = self.results[k]
            plain = CROSS if r.cross_score > r.x_score else X
            self.assertNotEqual(plain, r.expected, k)


class BenchmarkTest(unittest.TestCase):
    def test_measure_positive(self):
        self.assertGreater(measure_ms(mac_2d, CROSS3, X3), 0)

    def test_table_rows(self):
        rows = benchmark_sizes()
        self.assertEqual([r["n"] for r in rows], [3, 5, 13, 25])
        self.assertEqual([r["ops"] for r in rows], [9, 25, 169, 625])


if __name__ == "__main__":
    unittest.main()
