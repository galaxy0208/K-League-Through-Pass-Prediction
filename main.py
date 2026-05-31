import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 0) 파일 경로 / 옵션
# =========================
RAW_PATH = "raw_data.xlsx"
MAX_ROWS = 120_000  # 느리면 80_000~150_000
SEED = 42

# 경기장 스케일
PITCH_X = 105.0
PITCH_Y = 68.0

# 침투 패스(Through Pass) 프록시 정의 파라미터
MIN_PROGRESS = 12.0  # end_x - start_x 최소 전진(m)
MIN_ENDX = 50.0  # 최소 도착 x (전진 패스 느낌 강화)
ONLY_SUCCESS = True  # 성공 패스만

# 시퀀스 구성
K = 3  # 직전 K개의 패스(같은 팀, 같은 경기/하프)
MAX_GAP_SEC = 15  # 직전 패스들과 시간 간격 제한(초)

# Grid 분류
GRID_NX = 21
GRID_NY = 14


# =========================
# 1) 유틸: 경기장 플롯
# =========================
def draw_pitch():
    plt.plot([0, PITCH_X, PITCH_X, 0, 0], [0, 0, PITCH_Y, PITCH_Y, 0], "k-", lw=2)
    plt.plot([PITCH_X / 2, PITCH_X / 2], [0, PITCH_Y], "k--", lw=1)
    plt.xlim(-2, PITCH_X + 2)
    plt.ylim(-2, PITCH_Y + 2)


def save_show(fig_path=None):
    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=220)
    plt.show()


# =========================
# 2) 침투 패스 필터링
# =========================
def filter_through_passes(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Pass만
    d = d[d["type_name"] == "Pass"].copy()

    # 성공 Pass만 (옵션)
    if ONLY_SUCCESS:
        d = d[d["result_name"] == "Successful"].copy()

    # 전진성 + 도착 x 조건
    progress = (d["end_x"] - d["start_x"]).astype(float)
    d = d[progress >= MIN_PROGRESS].copy()
    d = d[d["end_x"] >= MIN_ENDX].copy()

    # 좌표 범위
    d = d[
        (d["start_x"].between(0, PITCH_X))
        & (d["end_x"].between(0, PITCH_X))
        & (d["start_y"].between(0, PITCH_Y))
        & (d["end_y"].between(0, PITCH_Y))
    ].copy()

    return d


# =========================
# 3) Grid 라벨링
# =========================
def xy_to_grid_class(x, y, nx=GRID_NX, ny=GRID_NY):
    gx = int(np.floor(x / PITCH_X * nx))
    gy = int(np.floor(y / PITCH_Y * ny))
    gx = min(max(gx, 0), nx - 1)
    gy = min(max(gy, 0), ny - 1)
    return gy * nx + gx


def class_to_cell_center(class_id, nx=GRID_NX, ny=GRID_NY):
    gx = int(class_id) % nx
    gy = int(class_id) // nx
    cx = (gx + 0.5) * (PITCH_X / nx)
    cy = (gy + 0.5) * (PITCH_Y / ny)
    return cx, cy


# =========================
# 4) 시퀀스 샘플 생성
# =========================
def build_sequence_dataset(raw_all: pd.DataFrame, through_df: pd.DataFrame):
    # 컨텍스트는 패스만 사용
    passes = raw_all[raw_all["type_name"] == "Pass"].copy()
    passes = passes.sort_values(
        ["game_id", "period_id", "time_seconds", "action_id"]
    ).reset_index(drop=True)

    # through_df에서 해당 패스 row index 찾기 위해 merge
    key_cols = [
        "game_id",
        "period_id",
        "time_seconds",
        "action_id",
        "player_id",
        "team_id",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
    ]
    tmp_passes = passes.reset_index().rename(columns={"index": "pass_idx"})
    merged = through_df.merge(
        tmp_passes[key_cols + ["pass_idx"]], on=key_cols, how="inner"
    )

    # numpy 캐시
    px = passes["start_x"].to_numpy()
    py = passes["start_y"].to_numpy()
    ex = passes["end_x"].to_numpy()
    ey = passes["end_y"].to_numpy()
    tt = passes["time_seconds"].to_numpy()
    gid = passes["game_id"].to_numpy()
    pid = passes["period_id"].to_numpy()
    tid = passes["team_id"].to_numpy()

    X_list, y_list, meta_list = [], [], []

    norm_dist = float(np.sqrt(PITCH_X**2 + PITCH_Y**2))

    for r in merged.itertuples(index=False):
        idx = int(r.pass_idx)
        g = r.game_id
        p = r.period_id
        team = r.team_id
        t_now = r.time_seconds

        # 직전 K개 패스 추적
        j = idx - 1
        seq = []
        while j >= 0 and len(seq) < K:
            # 경기/하프 바뀌면 중단
            if gid[j] != g or pid[j] != p:
                break
            # 같은 팀 패스만
            if tid[j] != team:
                j -= 1
                continue
            # 시간 간격 제한
            if (t_now - tt[j]) > MAX_GAP_SEC:
                break

            prog = max(ex[j] - px[j], 0.0)
            dist = float(np.sqrt((ex[j] - px[j]) ** 2 + (ey[j] - py[j]) ** 2))

            seq.append(
                [
                    px[j] / PITCH_X,
                    py[j] / PITCH_Y,
                    ex[j] / PITCH_X,
                    ey[j] / PITCH_Y,
                    prog / PITCH_X,
                    dist / norm_dist,
                ]
            )
            j -= 1

        if len(seq) < K:
            continue

        seq = seq[::-1]  # 과거 -> 현재
        x_vec = np.array(seq, dtype=float).reshape(-1)  # (K*6,)
        target_class = xy_to_grid_class(r.end_x, r.end_y)

        X_list.append(x_vec)
        y_list.append(int(target_class))
        meta_list.append(
            {
                "game_id": g,
                "period_id": p,
                "team_id": team,
                "target_end_x": float(r.end_x),
                "target_end_y": float(r.end_y),
            }
        )

    X = np.vstack(X_list) if X_list else np.empty((0, K * 6))
    y = np.array(y_list, dtype=int) if y_list else np.array([], dtype=int)
    meta = pd.DataFrame(meta_list) if meta_list else pd.DataFrame()
    return X, y, meta


# =========================
# 5) 히트맵 유틸
# =========================
def plot_grid_heatmap(grid_values, title, out_path=None):
    plt.figure(figsize=(10, 6.5))
    draw_pitch()
    plt.imshow(
        grid_values,
        origin="lower",
        extent=[0, PITCH_X, 0, PITCH_Y],
        aspect="auto",
        alpha=0.85,
    )
    plt.colorbar(label="Score / Probability")
    plt.title(title)
    save_show(out_path)


def actual_end_heatmap(through_df, nx=GRID_NX, ny=GRID_NY):
    H = np.zeros((ny, nx), dtype=float)
    for x, y in zip(through_df["end_x"].to_numpy(), through_df["end_y"].to_numpy()):
        c = xy_to_grid_class(float(x), float(y), nx, ny)
        gx = c % nx
        gy = c // nx
        H[gy, gx] += 1.0
    return np.log1p(H)


def proba_to_heatmap(proba_vec, nx=GRID_NX, ny=GRID_NY):
    return proba_vec.reshape(ny, nx)


# =========================
# 6) MAIN
# =========================
def main():
    print("[1/6] Load data...")
    raw = pd.read_excel(RAW_PATH)

    need_cols = [
        "game_id",
        "period_id",
        "time_seconds",
        "action_id",
        "type_name",
        "result_name",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "dx",
        "dy",
        "player_id",
        "team_id",
    ]
    raw = raw.dropna(subset=need_cols).copy()

    raw = raw.sort_values(
        ["game_id", "period_id", "time_seconds", "action_id"]
    ).reset_index(drop=True)

    if MAX_ROWS and len(raw) > MAX_ROWS:
        raw = raw.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)
        raw = raw.sort_values(
            ["game_id", "period_id", "time_seconds", "action_id"]
        ).reset_index(drop=True)

    print("Rows used:", len(raw))

    print("[2/6] Filter through passes...")
    through = filter_through_passes(raw)
    print("Through pass candidates:", len(through))

    print("[3/6] Save GT heatmap...")
    H_actual = actual_end_heatmap(through, GRID_NX, GRID_NY)
    plot_grid_heatmap(
        H_actual,
        "Ground Truth: Through Pass End Location (log count)",
        out_path="through_actual_heatmap.png",
    )

    print("[4/6] Build sequence dataset...")
    X, y, meta = build_sequence_dataset(raw, through)
    print("Sequence samples:", X.shape, y.shape)

    if len(y) < 200:
        print(
            "[warn] Too few samples for training. Try lowering MIN_PROGRESS or increasing MAX_GAP_SEC / MAX_ROWS."
        )
        print("Saved: through_actual_heatmap.png")
        return

    # 모델 선택 (AI 학습)
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200, n_jobs=-1)
        model_name = "LogisticRegression"
    except Exception:
        from sklearn.neural_network import MLPClassifier

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=50, random_state=SEED
        )
        model_name = "MLPClassifier"

    # stratify 조건부 적용 (희귀 클래스 있으면 비활성)
    class_counts = pd.Series(y).value_counts()
    too_rare = class_counts[class_counts < 2]
    strat = y if len(too_rare) == 0 else None
    if strat is None:
        print(f"[warn] rare classes (<2 samples): {len(too_rare)} -> stratify disabled")

    print("[5/6] Train AI model...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=strat
    )

    print(f"Training {model_name} ...")
    model.fit(X_train, y_train)

    # 검증(참고용)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_val)
        y_pred = np.argmax(proba, axis=1)
    else:
        y_pred = model.predict(X_val)
        proba = None

    acc = accuracy_score(y_val, y_pred)
    print("Val accuracy (ref):", round(acc, 4))

    # 예측 히트맵 저장 (predict_proba가 있을 때)
    if proba is not None:
        print("[6/6] Save predicted heatmap + scatter...")
        all_classes = np.arange(GRID_NX * GRID_NY)
        proba_full = np.zeros((proba.shape[0], len(all_classes)), dtype=float)

        for j, c in enumerate(model.classes_):
            proba_full[:, int(c)] = proba[:, j]

        mean_proba = proba_full.mean(axis=0)
        H_pred = proba_to_heatmap(mean_proba, GRID_NX, GRID_NY)
        plot_grid_heatmap(
            H_pred,
            "Predicted: Through Pass End Location (mean probability)",
            out_path="through_pred_heatmap.png",
        )

        # 비교 산점도(셀 중심점)
        n_show = min(200, len(y_val))
        rng = np.random.RandomState(SEED)
        idxs = rng.choice(len(y_val), size=n_show, replace=False)

        true_xy = np.array([class_to_cell_center(int(c)) for c in y_val[idxs]])
        pred_xy = np.array([class_to_cell_center(int(c)) for c in y_pred[idxs]])

        plt.figure(figsize=(10, 6.5))
        draw_pitch()
        plt.scatter(
            true_xy[:, 0], true_xy[:, 1], alpha=0.6, s=25, label="Actual (cell center)"
        )
        plt.scatter(
            pred_xy[:, 0], pred_xy[:, 1], alpha=0.6, s=25, label="Pred (cell center)"
        )
        plt.title("Actual vs Predicted (Grid Cell Centers) - Demo")
        plt.legend()
        save_show("through_pred_vs_actual.png")

        print("\nSaved files:")
        print("- through_actual_heatmap.png")
        print("- through_pred_heatmap.png")
        print("- through_pred_vs_actual.png")
    else:
        print("[warn] model has no predict_proba(); skipping predicted heatmap.")
        print("\nSaved files:")
        print("- through_actual_heatmap.png")


if __name__ == "__main__":
    main()
