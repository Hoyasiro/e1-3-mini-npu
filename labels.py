"""라벨 정규화(표준화).

data.json에는 같은 뜻의 라벨이 여러 모양으로 들어 있다.
  - expected 값: '+' 또는 'x'
  - filter 키  : 'cross' 또는 'x'
프로그램 안에서는 표준 라벨 두 개(CROSS='Cross', X='X')만 쓴다.
비교·출력 전에 반드시 normalize_label()을 거치게 해서
'+'와 'cross'를 다른 값으로 보는 실수를 막는다.
"""

CROSS = "Cross"
X = "X"
STANDARD_LABELS = (CROSS, X)

# 앞뒤 공백을 지우고 소문자로 바꾼 뒤 이 표에서 찾는다.
_ALIASES = {
    "+": CROSS,
    "cross": CROSS,
    "x": X,
}


def normalize_label(raw):
    """원본 라벨을 표준 라벨로 바꾼다. 해석할 수 없으면 None."""
    if not isinstance(raw, str):
        return None
    return _ALIASES.get(raw.strip().lower())
