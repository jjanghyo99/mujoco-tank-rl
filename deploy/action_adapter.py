# action_adapter.py

# 학습이 끝나고 Flask 서버 코드를 짤 때, 
# "정책이 출력한 [v, w] 값을 실제 API가 요구하는 
# {"moveWS": ..., "moveAD": ...} JSON 형태로 바꾸는 함수"가 필요할 텐데, 
# 그 부분만 미리 분리해서 짜둔 것

def action_to_command(v: float, w: float) -> dict:
    """정책 출력 [v, w] (-1~1) -> Tank Challenge API 포맷"""
    if abs(v) > 0.05:
        moveWS = {"command": "W" if v >= 0 else "S",
                  "weight": round(max(0.1, min(1.0, abs(v))), 2)}
    else:
        moveWS = {"command": "STOP", "weight": 0.1}

    if abs(w) > 0.05:
        moveAD = {"command": "D" if w >= 0 else "A",
                  "weight": round(max(0.1, min(1.0, abs(w))), 2)}
    else:
        moveAD = {"command": "STOP", "weight": 0.1}

    return {
        "moveWS": moveWS,
        "moveAD": moveAD,
        "turretQE": {"command": "STOP", "weight": 0.1},
        "turretRF": {"command": "STOP", "weight": 0.1},
        "fire": False,
    }