#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vanilla CFR (Counterfactual Regret Minimization) の
Kuhn Poker 実装 – Justin Sermeno 氏のブログ記事より再構成
-----------------------------------------------------------------
このスクリプトは二人零和不完全情報ゲーム『Kuhn Poker』に
Vanilla CFR を適用し、近似ナッシュ均衡戦略を学習します。

主な変更点
-----------
* 記事中の複数コード片を一つのファイルにまとめました。
* 日本語で詳細なコメントを追加し、数式との対応関係を明記しました。
* 変数名・ロジックは記事を忠実に踏襲し、可読性を最優先にしています。
* NumPy 以外の外部依存はありません（標準 Python 3 + NumPy）。

実行方法
---------
$ python cfr_kuhn_vanilla.py
（iterations の既定値は 10,000。時間と精度のトレードオフで調整可）
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# ──────────────────────────────────────────────────────────────
# 定数定義
# ──────────────────────────────────────────────────────────────
_N_ACTIONS = 2        # 各情報集合で取り得る行動数（check=0, bet=1）
_N_CARDS   = 3        # Kuhn Poker のカード枚数（J, Q, K）

# ──────────────────────────────────────────────────────────────
# メインループ
# ──────────────────────────────────────────────────────────────
def main() -> None:
    """
    CFR を指定回数反復し、期待利得と平均戦略を表示する。
    ----------------------------------------------------------------
    i_map : dict[str, InformationSet]
        文字列キー（"J rr" 等）を情報集合オブジェクトに対応付ける辞書。
    """
    i_map: dict[str, InformationSet] = {}
    n_iterations = 10_000           # 反復回数（精度が欲しければ増やす）

    expected_game_value = 0.0       # 反復ごとのゲーム値を累積
    ev_history: list[float] = []    # プレイヤー1の期待利得推移

    for t in range(1, n_iterations + 1):
        expected_game_value += cfr(i_map)      # 1 回分の木探索 → 利得を加算

        # 1 反復が終わったら全情報集合で次回戦略を計算
        for info_set in i_map.values():
            info_set.next_strategy()

        ev_history.append(expected_game_value / t)

    # 反復平均を取ることで近似ゲーム値を得る
    expected_game_value /= n_iterations

    display_results(expected_game_value, i_map)
    plot_ev_history(ev_history)

# ──────────────────────────────────────────────────────────────
# CFR 本体 – 深さ優先でゲーム木を全探索（Vanilla CFR）
# ──────────────────────────────────────────────────────────────
def cfr(
    i_map: dict[str, "InformationSet"],
    history: str = "",        # これまでの行動履歴： 'r','c','b' の列
    card_1: int = -1,         # プレイヤー1 の手札 (0:J,1:Q,2:K)
    card_2: int = -1,         # プレイヤー2 の手札
    pr_1: float = 1.0,        # プレイヤー1 の到達確率寄与 π^σ_1(h)
    pr_2: float = 1.0,        # プレイヤー2 の到達確率寄与 π^σ_2(h)
    pr_c: float = 1.0         # チャンス節（配札）の確率寄与 π^σ_c(h)
) -> float:
    """
    Counterfactual Regret Minimization アルゴリズム（再帰呼び出し）。
    ----------------------------------------------------------------
    戻り値
    ------
    util : float
        現在の手番プレイヤーにとっての期待利得。
    """

    # ① チャンスノード（配札）か？
    if is_chance_node(history):
        return chance_util(i_map)

    # ② 終端ノード（勝敗確定）か？
    if is_terminal(history):
        return terminal_util(history, card_1, card_2)

    # ③ 決定ノード（情報集合） – ここが CFR の核心
    is_player_1 = (len(history) % 2 == 0)      # 偶数手目なら P1 の番
    # 情報集合の取得（なければ生成）
    info_set = get_info_set(
        i_map,
        card_1 if is_player_1 else card_2,
        history
    )

    strategy = info_set.strategy               # 現在反復の戦略 σ_t(I)
    # 到達確率 π_t^σ_i(I) を蓄積（平均戦略計算用）
    if is_player_1:
        info_set.reach_pr += pr_1
    else:
        info_set.reach_pr += pr_2

    # 各アクションの「反事実的」ユーティリティを格納
    action_utils = np.zeros(_N_ACTIONS)

    for i, action in enumerate(("c", "b")):
        next_history = history + action

        # 再帰呼び出しで得られる値は「相手視点」なので -1 倍
        if is_player_1:
            action_utils[i] = -cfr(
                i_map, next_history,
                card_1, card_2,
                pr_1 * strategy[i], pr_2, pr_c
            )
        else:
            action_utils[i] = -cfr(
                i_map, next_history,
                card_1, card_2,
                pr_1, pr_2 * strategy[i], pr_c
            )

    # 情報集合全体の期待利得
    util = float(np.dot(action_utils, strategy))

    # 後悔値 Δ = u(a) - ū を計算
    regrets = action_utils - util
    # Counterfactual Regret を累積 – π_{-i}^σ * π_c^σ を掛ける点に注意
    if is_player_1:
        info_set.regret_sum += pr_2 * pr_c * regrets
    else:
        info_set.regret_sum += pr_1 * pr_c * regrets

    return util

# ──────────────────────────────────────────────────────────────
#   ゲーム木ユーティリティ関連関数
# ──────────────────────────────────────────────────────────────
def is_chance_node(history: str) -> bool:
    """配札前（履歴が空文字列）ならチャンスノード。"""
    return history == ""

def chance_util(i_map: dict[str, "InformationSet"]) -> float:
    """配札の全組み合わせ（6 通り）を列挙し期待値を平均。"""
    expected_value = 0.0
    n_possibilities = 6          # 3P2 = 6

    for i in range(_N_CARDS):
        for j in range(_N_CARDS):
            if i != j:           # 同じカードは引けない
                expected_value += cfr(
                    i_map, "rr", i, j,
                    1.0, 1.0, 1.0 / n_possibilities
                )

    return expected_value / n_possibilities

def is_terminal(history: str) -> bool:
    """終端履歴集合かどうかの判定（ハードコード）。"""
    return history in {
        "rrcc",  # チェック–チェック
        "rrcbc", # C–B–C → フォールド
        "rrcbb", # C–B–B → ショーダウン
        "rrbc",  # B–C → フォールド
        "rrbb",  # B–B → ショーダウン
    }

def terminal_util(history: str, card_1: int, card_2: int) -> int:
    """
    勝敗に応じて利得を返す。
    チップ単位：フォールド = +1、ノーベット勝負 = ±1、1 ベット勝負 = ±2
    """
    # 偶数長なら最後に動いたのは P2、したがって手番は P1
    card_player    = card_1 if len(history) % 2 == 0 else card_2
    card_opponent  = card_2 if len(history) % 2 == 0 else card_1

    if history in ("rrcbc", "rrbc"):          # 相手フォールド
        return 1
    if history == "rrcc":                     # チェック–チェック
        return 1 if card_player > card_opponent else -1
    # ベットが入った後のショーダウン（pot = 4）
    return 2 if card_player > card_opponent else -2

# ──────────────────────────────────────────────────────────────
#   情報集合ユーティリティ
# ──────────────────────────────────────────────────────────────
def card_str(card: int) -> str:
    """整数表現を 'J','Q','K' の文字列へ変換。"""
    return ("J", "Q", "K")[card]

def describe_history(history: str) -> str:
    """履歴文字列を日本語の人間可読形式に変換する。"""
    # 'rr' は配札を表すダミーなので無視
    actions = []
    bet_happened = False
    # 先手は P1, 次手は P2 ... と交互
    for i, ch in enumerate(history[2:]):
        actor = "P1" if i % 2 == 0 else "P2"
        if ch == "c":
            action = "チェック" if not bet_happened else "フォールド"
        else:  # 'b'
            action = "ベット" if not bet_happened else "コール"
            if not bet_happened:
                bet_happened = True
        actions.append(f"{actor}:{action}")

    return "開始直後" if not actions else "→".join(actions)

def get_info_set(
    i_map: dict[str, "InformationSet"],
    card: int,
    history: str
) -> "InformationSet":
    """
    カード + 履歴 で一意になるキーを生成し、対応する情報集合を返す。
    存在しなければ新規作成して dict に登録。
    """
    key = f"{card_str(card)} {history}"
    if key not in i_map:
        i_map[key] = InformationSet(key)
    return i_map[key]

class InformationSet:
    """
    CFR が保持する各情報集合 I のデータ構造。
    --------------------------------------------------------------
    regret_sum    : 行動別正味後悔値 ∑ R_t(I,a)
    strategy_sum  : 行動別到達確率付き戦略 ∑ π_i^t(I) σ_t(I,a)
    strategy      : 現反復で用いる戦略 σ_t(I,a)
    reach_pr      : その反復での到達確率 π_i^t(I) を一時的に蓄積
    reach_pr_sum  : 累積到達確率 ∑ π_i^t(I)（平均戦略の正規化用）
    """
    def __init__(self, key: str) -> None:
        self.key = key
        self.regret_sum   = np.zeros(_N_ACTIONS)
        self.strategy_sum = np.zeros(_N_ACTIONS)
        self.strategy     = np.repeat(1 / _N_ACTIONS, _N_ACTIONS)
        self.reach_pr     = 0.0
        self.reach_pr_sum = 0.0

    # ─── 反復終了時に呼び出し ──────────────────────────────
    def next_strategy(self) -> None:
        """現在戦略を strategy_sum に加算し、新戦略を計算。"""
        self.strategy_sum  += self.reach_pr * self.strategy
        self.strategy       = self._calc_strategy()
        self.reach_pr_sum  += self.reach_pr
        self.reach_pr       = 0.0        # 次反復に備えてリセット

    # ─── Regret Matching で次戦略を決定 ──────────────────────
    def _calc_strategy(self) -> np.ndarray:
        """
        Regret Matching:
            正の後悔値のみを考慮し比例配分。
            全後悔値 ≤ 0 の場合は一様戦略。
        """
        positive_regrets = np.maximum(self.regret_sum, 0.0)
        total = positive_regrets.sum()
        if total > 0:
            return positive_regrets / total
        # 後悔値が全て負または 0 → ランダムに一様行動
        return np.repeat(1 / _N_ACTIONS, _N_ACTIONS)

    # ─── 収束後に Nash 近似戦略を取得 ─────────────────────
    def get_average_strategy(self) -> np.ndarray:
        """
        平均戦略 \bar{σ}(I,a) を計算。
        reach_pr_sum = 0 になることは理論上ないが念のため保護。
        """
        avg = np.divide(
            self.strategy_sum,
            self.reach_pr_sum if self.reach_pr_sum > 0 else 1.0
        )
        # Purification: 誤差レベル (<0.1%) を 0 扱いし再正規化
        avg[np.where(avg < 0.001)] = 0.0
        total = avg.sum()
        return avg / total if total > 0 else np.repeat(1 / _N_ACTIONS, _N_ACTIONS)

    # ─── デバッグ用表示 ─────────────────────────────────────
    def __str__(self) -> str:
        card, history = self.key.split()
        readable = describe_history(history)
        probs = [f"{p:0.2f}" for p in self.get_average_strategy()]
        prob_str = "/".join(probs)
        return f"カード{card} | 履歴: {readable} | 戦略: {prob_str}"

# ──────────────────────────────────────────────────────────────
# 結果表示
# ──────────────────────────────────────────────────────────────
def display_results(ev: float, i_map: dict[str, InformationSet]) -> None:
    """期待利得と両者の平均戦略を人間可読で出力。"""
    print(f"player 1 expected value: {ev}")
    print(f"player 2 expected value: {-ev}\n")

    print("player 1 strategies:")
    for _, v in sorted(filter(lambda kv: len(kv[0]) % 2 == 0, i_map.items())):
        print(v)

    print("\nplayer 2 strategies:")
    for _, v in sorted(filter(lambda kv: len(kv[0]) % 2 == 1, i_map.items())):
        print(v)

def plot_ev_history(ev_history: list[float]) -> None:
    """反復ごとの期待利得推移をグラフ表示する。"""
    iterations = np.arange(1, len(ev_history) + 1)

    plt.figure()
    plt.plot(iterations, ev_history, label="プレイヤー1")
    plt.plot(iterations, -np.array(ev_history), label="プレイヤー2")
    plt.xlabel("反復回数")
    plt.ylabel("期待利得")
    plt.title("期待利得の変化")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────
# エントリポイント
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
