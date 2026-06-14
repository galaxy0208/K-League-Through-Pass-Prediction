# -*- coding: utf-8 -*-
"""
LANCON: Pathfinder - LSTM / Time-Series Pipeline (Phase 0 + Phase 1)
가변 길이 'possession 시퀀스'를 LSTM으로 인코딩해 다음 침투 패스 도착 공간의
확률 분포(히트맵)를 예측한다.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

# =========================
# 0) 설정
# =========================
RAW_PATH = "raw_data.csv"
MATCH_PATH = "match_info.csv"
SEED = 42

PITCH_X, PITCH_Y = 105.0, 68.0
NORM_DIST = float(np.sqrt(PITCH_X ** 2 + PITCH_Y ** 2))

# 침투 패스 프록시 정의
MIN_PROGRESS = 12.0
MIN_ENDX = 50.0

# 시퀀스/시계열 파라미터
MAX_GAP_SEC = 15.0      # possession 분절 기준 (이벤트 간격)
MAX_SEQ_LEN = 20        # LSTM 입력 최대 길이 (직전 이벤트 수)
MIN_SEQ_LEN = 2         # 최소 빌드업 길이

# Grid
GRID_NX, GRID_NY = 21, 14
N_CLASS = GRID_NX * GRID_NY

# 학습
BATCH = 256
EPOCHS = 12
LR = 1e-3
HIDDEN = 128
EMB_POS = 8
EMB_TYPE = 8
SOFT_SIGMA = 5.0        # 소프트 타깃 가우시안 표준편차(m)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 빠른 검증용: None 이면 전체 게임 사용
MAX_GAMES = None

np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# 1) Grid 유틸
# =========================
def xy_to_grid_class(x, y, nx=GRID_NX, ny=GRID_NY):
    gx = min(max(int(np.floor(x / PITCH_X * nx)), 0), nx - 1)
    gy = min(max(int(np.floor(y / PITCH_Y * ny)), 0), ny - 1)
    return gy * nx + gx


def class_to_cell_center(class_id, nx=GRID_NX, ny=GRID_NY):
    gx, gy = int(class_id) % nx, int(class_id) // nx
    return (gx + 0.5) * (PITCH_X / nx), (gy + 0.5) * (PITCH_Y / ny)


# 모든 셀 중심 좌표 (N_CLASS, 2)
_CELL_CENTERS = np.array([class_to_cell_center(c) for c in range(N_CLASS)], dtype=np.float32)


def draw_pitch(ax=None):
    ax = ax or plt.gca()
    ax.plot([0, PITCH_X, PITCH_X, 0, 0], [0, 0, PITCH_Y, PITCH_Y, 0], "k-", lw=2)
    ax.plot([PITCH_X / 2, PITCH_X / 2], [0, PITCH_Y], "k--", lw=1)
    ax.set_xlim(-2, PITCH_X + 2)
    ax.set_ylim(-2, PITCH_Y + 2)


# =========================
# 2) 데이터 로드 + 시계열 전처리
# =========================
def load_data():
    raw = pd.read_csv(RAW_PATH)
    need = ["game_id", "period_id", "time_seconds", "action_id",
            "type_name", "result_name", "start_x", "start_y",
            "end_x", "end_y", "team_id"]
    raw = raw.dropna(subset=need).copy()
    raw = raw.sort_values(["game_id", "period_id", "time_seconds", "action_id"]).reset_index(drop=True)

    if MAX_GAMES is not None:
        keep = raw["game_id"].drop_duplicates().head(MAX_GAMES)
        raw = raw[raw["game_id"].isin(keep)].reset_index(drop=True)

    # 좌표 범위 클리핑(노이즈 방지)
    raw = raw[(raw["start_x"].between(0, PITCH_X)) & (raw["end_x"].between(0, PITCH_X)) &
              (raw["start_y"].between(0, PITCH_Y)) & (raw["end_y"].between(0, PITCH_Y))].reset_index(drop=True)

    # 결측 카테고리 처리
    raw["main_position"] = raw["main_position"].fillna("UNK") if "main_position" in raw else "UNK"
    raw["type_name"] = raw["type_name"].fillna("UNK")

    return raw


def add_running_goal_diff(raw):
    """Goal 이벤트로부터 각 이벤트 시점의 (해당 팀 기준) 누적 득실차를 계산한다. (비누수)"""
    raw = raw.copy()
    is_goal = (raw["result_name"] == "Goal").astype(int)
    # 팀별 누적 골 (해당 이벤트 발생 '전' 상태를 쓰기 위해 shift)
    raw["_goal"] = is_goal
    raw["team_cum_goal"] = (
        raw.groupby(["game_id", "team_id"])["_goal"].cumsum() - raw["_goal"]
    )
    # 경기별 전체 누적 골
    tot = raw.groupby("game_id")["_goal"].cumsum() - raw["_goal"]
    raw["game_cum_goal"] = tot
    raw["opp_cum_goal"] = raw["game_cum_goal"] - raw["team_cum_goal"]
    raw["goal_diff"] = raw["team_cum_goal"] - raw["opp_cum_goal"]
    raw.drop(columns=["_goal", "team_cum_goal", "game_cum_goal", "opp_cum_goal"], inplace=True)
    return raw


def assign_possession(raw):
    """team/period/game 변경 또는 시간 간격 초과 시 새 possession 으로 분절."""
    raw = raw.copy()
    prev_team = raw["team_id"].shift(1)
    prev_period = raw["period_id"].shift(1)
    prev_game = raw["game_id"].shift(1)
    prev_t = raw["time_seconds"].shift(1)
    gap = raw["time_seconds"] - prev_t

    new_poss = (
        (raw["team_id"] != prev_team)
        | (raw["period_id"] != prev_period)
        | (raw["game_id"] != prev_game)
        | (gap > MAX_GAP_SEC)
        | (gap < 0)
    )
    raw["poss_id"] = new_poss.cumsum()
    # 이전 이벤트와의 dt (possession 내부에서만 의미)
    raw["dt"] = gap.clip(lower=0, upper=MAX_GAP_SEC).fillna(0.0)
    raw.loc[new_poss, "dt"] = 0.0
    return raw


def is_through_pass(row):
    return (
        row["type_name"] == "Pass"
        and row["result_name"] == "Successful"
        and (row["end_x"] - row["start_x"]) >= MIN_PROGRESS
        and row["end_x"] >= MIN_ENDX
    )


# 연속 피처 빌더 (timestep 단위, 9차원)
def event_cont_features(sx, sy, ex, ey, dt, gdiff, tsec):
    prog = max(ex - sx, 0.0)
    dist = float(np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2))
    return [
        sx / PITCH_X, sy / PITCH_Y, ex / PITCH_X, ey / PITCH_Y,
        prog / PITCH_X, dist / NORM_DIST,
        dt / MAX_GAP_SEC,
        float(np.clip(gdiff, -3, 3)) / 3.0,
        float(np.clip(tsec, 0, 3000)) / 3000.0,
    ]
N_CONT = 9


def build_sequences(raw, pos2idx, type2idx):
    """각 침투 패스에 대해 같은 possession 내 직전 이벤트 시퀀스를 만든다."""
    cols = ["type_name", "result_name", "start_x", "start_y", "end_x", "end_y",
            "dt", "goal_diff", "time_seconds", "main_position",
            "game_id", "player_name_ko"]
    samples = []
    sx = raw["start_x"].to_numpy(); sy = raw["start_y"].to_numpy()
    ex = raw["end_x"].to_numpy();   ey = raw["end_y"].to_numpy()
    dt = raw["dt"].to_numpy();      gd = raw["goal_diff"].to_numpy()
    ts = raw["time_seconds"].to_numpy()
    tn = raw["type_name"].to_numpy(); rn = raw["result_name"].to_numpy()
    mp = raw["main_position"].astype(str).to_numpy()
    gid = raw["game_id"].to_numpy()
    pnm = raw["player_name_ko"].astype(str).to_numpy() if "player_name_ko" in raw else np.array(["?"] * len(raw))

    for _, sub in raw.groupby("poss_id", sort=False):
        idx = sub.index.to_numpy()
        if len(idx) < MIN_SEQ_LEN + 1:
            continue
        for pos_in_seq in range(MIN_SEQ_LEN, len(idx)):
            i = idx[pos_in_seq]
            # 침투 패스인 이벤트만 타깃으로
            if not (tn[i] == "Pass" and rn[i] == "Successful"
                    and (ex[i] - sx[i]) >= MIN_PROGRESS and ex[i] >= MIN_ENDX):
                continue
            hist = idx[max(0, pos_in_seq - MAX_SEQ_LEN):pos_in_seq]  # 직전 이벤트들
            if len(hist) < MIN_SEQ_LEN:
                continue
            cont = np.array([event_cont_features(sx[h], sy[h], ex[h], ey[h], dt[h], gd[h], ts[h]) for h in hist],
                            dtype=np.float32)
            pos_ids = np.array([pos2idx.get(mp[h], 0) for h in hist], dtype=np.int64)
            typ_ids = np.array([type2idx.get(tn[h], 0) for h in hist], dtype=np.int64)
            target = xy_to_grid_class(ex[i], ey[i])
            samples.append({
                "cont": cont, "pos": pos_ids, "typ": typ_ids,
                "target": target, "end_x": float(ex[i]), "end_y": float(ey[i]),
                "game_id": int(gid[i]), "player": pnm[i],
            })
    return samples


# =========================
# 3) Dataset / collate
# =========================
class SeqDataset(Dataset):
    def __init__(self, samples):
        self.s = samples
    def __len__(self):
        return len(self.s)
    def __getitem__(self, i):
        s = self.s[i]
        return (torch.from_numpy(s["cont"]),
                torch.from_numpy(s["pos"]),
                torch.from_numpy(s["typ"]),
                s["target"], s["end_x"], s["end_y"])


def collate(batch):
    conts, poss, typs, tgts, exs, eys = zip(*batch)
    lengths = torch.tensor([c.shape[0] for c in conts], dtype=torch.long)
    cont = pad_sequence(conts, batch_first=True)            # (B, L, N_CONT)
    pos = pad_sequence(poss, batch_first=True)              # (B, L)
    typ = pad_sequence(typs, batch_first=True)
    tgt = torch.tensor(tgts, dtype=torch.long)
    ex = torch.tensor(exs, dtype=torch.float32)
    ey = torch.tensor(eys, dtype=torch.float32)
    return cont, pos, typ, lengths, tgt, ex, ey


# =========================
# 4) LSTM 모델
# =========================
class LSTMPredictor(nn.Module):
    def __init__(self, n_pos, n_type):
        super().__init__()
        self.emb_pos = nn.Embedding(n_pos, EMB_POS) # 이산적인 범주형 정수 ID 데이터를 연속적인 벡터 공간으로 임베딩
        self.emb_typ = nn.Embedding(n_type, EMB_TYPE)
        self.lstm = nn.LSTM(N_CONT + EMB_POS + EMB_TYPE, HIDDEN, batch_first=True)
        # LSTM의 출력(Hidden State)을 받아 최종 클래스별 확률(Logits)로 변환하는 다층 퍼셉트론(MLP)
        # 과적합 방지를 위한 Dropout(0.2)과 비선형 활성화 함수 ReLU가 중간에 포함
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(HIDDEN, N_CLASS),
        )
    def forward(self, cont, pos, typ, lengths):
        x = torch.cat([cont, self.emb_pos(pos), self.emb_typ(typ)], dim=-1)
        # 실제 데이터가 있는 타임스텝만 콤팩트하게 압축하여 LSTM이 쓸데없는 패딩 연산을 건너뜀
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(h[-1])  # (B, N_CLASS) logits


# =========================
# 5) 소프트 타깃 (2D 가우시안) + 평가
# =========================
def soft_targets(target_classes):
    centers = _CELL_CENTERS                                  # (N_CLASS,2)
    tgt_xy = centers[target_classes]                         # (B,2)
    d2 = ((centers[None, :, :] - tgt_xy[:, None, :]) ** 2).sum(-1)  # (B,N_CLASS)
    w = np.exp(-d2 / (2 * SOFT_SIGMA ** 2))
    w /= w.sum(1, keepdims=True)
    return torch.from_numpy(w.astype(np.float32))


def topk_acc(logits, tgt, k):
    topk = logits.topk(k, dim=1).indices
    return (topk == tgt[:, None]).any(1).float().mean().item()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_tgt, all_ex, all_ey = [], [], [], []
    for cont, pos, typ, lengths, tgt, ex, ey in loader:
        logits = model(cont.to(DEVICE), pos.to(DEVICE), typ.to(DEVICE), lengths)
        all_logits.append(logits.cpu()); all_tgt.append(tgt)
        all_ex.append(ex); all_ey.append(ey)
    logits = torch.cat(all_logits); tgt = torch.cat(all_tgt)
    ex = torch.cat(all_ex).numpy(); ey = torch.cat(all_ey).numpy()
    proba = torch.softmax(logits, 1)
    pred_cls = logits.argmax(1).numpy()
    pred_xy = _CELL_CENTERS[pred_cls]
    dist = np.sqrt((pred_xy[:, 0] - ex) ** 2 + (pred_xy[:, 1] - ey) ** 2).mean()
    nll = -torch.log(proba[torch.arange(len(tgt)), tgt] + 1e-9).mean().item()
    return {
        "top1": topk_acc(logits, tgt, 1),
        "top3": topk_acc(logits, tgt, 3),
        "top5": topk_acc(logits, tgt, 5),
        "dist_m": float(dist),
        "nll": nll,
        "proba_mean": proba.mean(0).numpy(),
    }


# =========================
# 6) MAIN
# =========================
def main():
    print(f"[device] {DEVICE}")
    print("[1/6] Load + time-series preprocess...")
    raw = load_data()
    raw = add_running_goal_diff(raw)
    raw = assign_possession(raw)
    print(f"  rows={len(raw)}, games={raw['game_id'].nunique()}, possessions={raw['poss_id'].nunique()}")

    # 카테고리 사전
    positions = ["UNK"] + sorted(raw["main_position"].astype(str).unique().tolist())
    types = ["UNK"] + sorted(raw["type_name"].astype(str).unique().tolist())
    pos2idx = {p: i for i, p in enumerate(positions)}
    type2idx = {t: i for i, t in enumerate(types)}

    print("[2/6] Build possession sequences (targets = through passes)...")
    samples = build_sequences(raw, pos2idx, type2idx)
    print(f"  sequence samples: {len(samples)}")
    if len(samples) < 200:
        print("[warn] too few samples"); return

    # ---- 경기 단위 + 시간순 분할 (누수 방지) ----
    try:
        mi = pd.read_csv(MATCH_PATH)[["game_id", "game_date"]]
        order = mi.sort_values("game_date")["game_id"].tolist()
    except Exception:
        order = sorted({s["game_id"] for s in samples})
    games = [g for g in order if g in {s["game_id"] for s in samples}]
    n = len(games)
    train_g = set(games[: int(n * 0.7)])
    val_g = set(games[int(n * 0.7): int(n * 0.85)])
    test_g = set(games[int(n * 0.85):])

    def subset(gset):
        return [s for s in samples if s["game_id"] in gset]
    tr, va, te = subset(train_g), subset(val_g), subset(test_g)
    print(f"  split (games {len(train_g)}/{len(val_g)}/{len(test_g)}) -> samples {len(tr)}/{len(va)}/{len(te)}")

    tr_loader = DataLoader(SeqDataset(tr), batch_size=BATCH, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(SeqDataset(va), batch_size=BATCH, shuffle=False, collate_fn=collate)
    te_loader = DataLoader(SeqDataset(te), batch_size=BATCH, shuffle=False, collate_fn=collate)

    print("[3/6] Train LSTM...")
    model = LSTMPredictor(len(positions), len(types)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    kl = nn.KLDivLoss(reduction="batchmean")

    best_val, best_state = 1e9, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot = 0.0
        for cont, pos, typ, lengths, tgt, ex, ey in tr_loader:
            soft = soft_targets(tgt.numpy()).to(DEVICE)
            logits = model(cont.to(DEVICE), pos.to(DEVICE), typ.to(DEVICE), lengths)
            loss = kl(torch.log_softmax(logits, 1), soft)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(tgt)
        m = evaluate(model, va_loader)
        print(f"  ep{ep:02d} loss={tot/len(tr):.4f} | val top1={m['top1']:.3f} "
              f"top3={m['top3']:.3f} top5={m['top5']:.3f} dist={m['dist_m']:.2f}m nll={m['nll']:.3f}")
        if m["nll"] < best_val:
            best_val = m["nll"]; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    print("[4/6] Test evaluation (best model)...")
    mt = evaluate(model, te_loader)
    print(f"  TEST top1={mt['top1']:.3f} top3={mt['top3']:.3f} top5={mt['top5']:.3f} "
          f"dist={mt['dist_m']:.2f}m nll={mt['nll']:.3f}")

    # 기준선(빈도) 비교: 학습셋 타깃 분포의 top-1
    tr_counts = np.bincount([s["target"] for s in tr], minlength=N_CLASS)
    base_top1 = (np.array([s["target"] for s in te]) == tr_counts.argmax()).mean()
    print(f"  [baseline majority-cell top1] {base_top1:.3f}")

    print("[5/6] Save heatmaps...")
    # Ground truth (test)
    H_act = np.zeros((GRID_NY, GRID_NX))
    for s in te:
        c = s["target"]; H_act[c // GRID_NX, c % GRID_NX] += 1
    _heatmap(np.log1p(H_act), "Ground Truth: Through Pass End (test, log count)", "through_actual_heatmap.png")
    # Predicted mean proba
    H_pred = mt["proba_mean"].reshape(GRID_NY, GRID_NX)
    _heatmap(H_pred, "LSTM Predicted: mean probability (test)", "through_pred_heatmap.png")

    print("[6/6] Actual vs Predicted scatter...")
    _scatter_actual_vs_pred(model, te)
    print("\nSaved: through_actual_heatmap.png, through_pred_heatmap.png, through_pred_vs_actual.png")


def _heatmap(grid, title, out):
    plt.figure(figsize=(10, 6.5)); draw_pitch()
    plt.imshow(grid, origin="lower", extent=[0, PITCH_X, 0, PITCH_Y], aspect="auto", alpha=0.85)
    plt.colorbar(label="Score / Probability"); plt.title(title); plt.tight_layout()
    plt.savefig(out, dpi=180); plt.close()


@torch.no_grad()
def _scatter_actual_vs_pred(model, samples, n_show=200):
    model.eval()
    sel = samples[:n_show] if len(samples) <= n_show else list(np.random.choice(samples, n_show, replace=False))
    loader = DataLoader(SeqDataset(sel), batch_size=len(sel), collate_fn=collate)
    cont, pos, typ, lengths, tgt, ex, ey = next(iter(loader))
    logits = model(cont.to(DEVICE), pos.to(DEVICE), typ.to(DEVICE), lengths)
    pred_cls = logits.argmax(1).cpu().numpy()
    pred_xy = _CELL_CENTERS[pred_cls]
    plt.figure(figsize=(10, 6.5)); draw_pitch()
    plt.scatter(ex.numpy(), ey.numpy(), alpha=0.6, s=25, label="Actual end")
    plt.scatter(pred_xy[:, 0], pred_xy[:, 1], alpha=0.6, s=25, label="Pred cell center")
    plt.title("Actual vs LSTM Predicted (test sample)"); plt.legend(); plt.tight_layout()
    plt.savefig("through_pred_vs_actual.png", dpi=180); plt.close()


if __name__ == "__main__":
    main()
