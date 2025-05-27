import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

class RPSRegretMatching:
    """
    じゃんけんゲームのRegret Matchingアルゴリズム実装
    """
    
    def __init__(self):
        """
        初期化
        行動: 0=グー, 1=チョキ, 2=パー
        """
        self.num_actions = 3
        
        # じゃんけんのペイオフ行列
        # payoff_matrix[i][j] = プレイヤー1が行動i、プレイヤー2が行動jを取った時のプレイヤー1の利得
        self.payoff_matrix = np.array([
            [0, 1, -1],   # グー vs (グー, チョキ, パー)
            [-1, 0, 1],   # チョキ vs (グー, チョキ, パー)
            [1, -1, 0]    # パー vs (グー, チョキ, パー)
        ])
        
    def get_strategy_from_regrets(self, regrets):
        """
        累積後悔値から混合戦略を計算
        
        Parameters:
        regrets (np.array): 各行動の累積後悔値
        
        Returns:
        np.array: 混合戦略（確率分布）
        """
        # 正の後悔値のみを考慮
        positive_regrets = np.maximum(regrets, 0)
        
        # 全ての後悔値が0の場合は一様分布
        if np.sum(positive_regrets) == 0:
            return np.ones(self.num_actions) / self.num_actions
        
        # 後悔値に比例した確率分布を返す
        return positive_regrets / np.sum(positive_regrets)
    
    def best_response_regret_matching(self, opponent_strategy, iterations=10000):
        """
        相手の混合戦略を固定したときの最適応答戦略をRegret Matchingで求める
        
        Parameters:
        opponent_strategy (np.array): 相手の混合戦略
        iterations (int): 反復回数
        
        Returns:
        np.array: 最適応答戦略
        """
        # 累積後悔値を初期化
        cumulative_regrets = np.zeros(self.num_actions)
        
        # 平均戦略を追跡
        strategy_sum = np.zeros(self.num_actions)
        
        for t in range(iterations):
            # 現在の戦略を計算
            strategy = self.get_strategy_from_regrets(cumulative_regrets)
            strategy_sum += strategy
            
            # 相手の行動をサンプリング
            opponent_action = np.random.choice(self.num_actions, p=opponent_strategy)
            
            # 各行動の期待利得を計算
            action_values = self.payoff_matrix[:, opponent_action]
            
            # 実際に選択した行動
            my_action = np.random.choice(self.num_actions, p=strategy)
            
            # 実際に得た利得
            actual_value = action_values[my_action]
            
            # 各行動の後悔値を計算（他の行動を選んでいたら得られた利得 - 実際の利得）
            regrets = action_values - actual_value
            
            # 累積後悔値を更新
            cumulative_regrets += regrets
        
        # 平均戦略を返す
        return strategy_sum / np.sum(strategy_sum)
    
    def nash_equilibrium_regret_matching(self, iterations=100000):
        """
        双方のプレイヤーでRegret Matchingを行ってナッシュ均衡を近似
        
        Parameters:
        iterations (int): 反復回数
        
        Returns:
        tuple: (プレイヤー1の均衡戦略, プレイヤー2の均衡戦略)
        """
        # 各プレイヤーの累積後悔値を初期化
        regrets_p1 = np.zeros(self.num_actions)
        regrets_p2 = np.zeros(self.num_actions)
        
        # 平均戦略を追跡
        strategy_sum_p1 = np.zeros(self.num_actions)
        strategy_sum_p2 = np.zeros(self.num_actions)
        
        # 収束過程を記録
        history_p1 = []
        history_p2 = []
        
        for t in range(iterations):
            # 各プレイヤーの現在の戦略を計算
            strategy_p1 = self.get_strategy_from_regrets(regrets_p1)
            strategy_p2 = self.get_strategy_from_regrets(regrets_p2)
            
            # 平均戦略を更新
            strategy_sum_p1 += strategy_p1
            strategy_sum_p2 += strategy_p2
            
            # 各プレイヤーの行動をサンプリング
            action_p1 = np.random.choice(self.num_actions, p=strategy_p1)
            action_p2 = np.random.choice(self.num_actions, p=strategy_p2)
            
            # プレイヤー1の後悔値を計算
            values_p1 = self.payoff_matrix[:, action_p2]
            actual_value_p1 = values_p1[action_p1]
            regrets_p1 += values_p1 - actual_value_p1
            
            # プレイヤー2の後悔値を計算（プレイヤー2の利得は-プレイヤー1の利得）
            values_p2 = -self.payoff_matrix[action_p1, :]
            actual_value_p2 = values_p2[action_p2]
            regrets_p2 += values_p2 - actual_value_p2
            
            # 定期的に進捗を記録
            if t % 1000 == 0:
                avg_strategy_p1 = strategy_sum_p1 / np.sum(strategy_sum_p1)
                avg_strategy_p2 = strategy_sum_p2 / np.sum(strategy_sum_p2)
                history_p1.append(avg_strategy_p1)
                history_p2.append(avg_strategy_p2)
        
        # 最終的な平均戦略を計算
        equilibrium_p1 = strategy_sum_p1 / np.sum(strategy_sum_p1)
        equilibrium_p2 = strategy_sum_p2 / np.sum(strategy_sum_p2)
        
        return equilibrium_p1, equilibrium_p2, history_p1, history_p2
    
    def verify_nash_equilibrium(self, strategy_p1, strategy_p2):
        """
        求めた戦略がナッシュ均衡に近いかを検証
        
        Parameters:
        strategy_p1 (np.array): プレイヤー1の戦略
        strategy_p2 (np.array): プレイヤー2の戦略
        
        Returns:
        dict: 検証結果
        """
        # プレイヤー1の期待利得
        expected_payoff_p1 = strategy_p1 @ self.payoff_matrix @ strategy_p2
        
        # プレイヤー1の各純戦略での利得
        pure_payoffs_p1 = self.payoff_matrix @ strategy_p2
        
        # プレイヤー1の最善応答
        best_response_p1 = np.max(pure_payoffs_p1)
        
        # プレイヤー2の期待利得（ゼロサムゲームなので-expected_payoff_p1）
        expected_payoff_p2 = -expected_payoff_p1
        
        # プレイヤー2の各純戦略での利得
        pure_payoffs_p2 = -strategy_p1 @ self.payoff_matrix
        
        # プレイヤー2の最善応答
        best_response_p2 = np.max(pure_payoffs_p2)
        
        return {
            'p1_expected_payoff': expected_payoff_p1,
            'p1_best_response': best_response_p1,
            'p1_regret': best_response_p1 - expected_payoff_p1,
            'p2_expected_payoff': expected_payoff_p2,
            'p2_best_response': best_response_p2,
            'p2_regret': best_response_p2 - expected_payoff_p2
        }


def main():
    """
    実行例
    """
    rps = RPSRegretMatching()
    
    print("=== じゃんけんゲームのRegret Matching ===\n")
    
    # 1. 相手の戦略を固定した場合の最適応答
    print("1. 相手の戦略を固定した場合の最適応答")
    print("-" * 40)
    
    # 相手がグーを多く出す戦略 (グー:0.6, チョキ:0.2, パー:0.2)
    opponent_strategy = np.array([0.6, 0.2, 0.2])
    print(f"相手の戦略: グー={opponent_strategy[0]:.1f}, チョキ={opponent_strategy[1]:.1f}, パー={opponent_strategy[2]:.1f}")
    
    best_response = rps.best_response_regret_matching(opponent_strategy)
    print(f"最適応答戦略: グー={best_response[0]:.3f}, チョキ={best_response[1]:.3f}, パー={best_response[2]:.3f}")
    print("(相手がグーを多く出すので、パーを多く出すのが最適)\n")
    
    # 2. ナッシュ均衡の計算
    print("2. ナッシュ均衡の計算")
    print("-" * 40)
    
    equilibrium_p1, equilibrium_p2, history_p1, history_p2 = rps.nash_equilibrium_regret_matching()
    
    print(f"プレイヤー1の均衡戦略: グー={equilibrium_p1[0]:.3f}, チョキ={equilibrium_p1[1]:.3f}, パー={equilibrium_p1[2]:.3f}")
    print(f"プレイヤー2の均衡戦略: グー={equilibrium_p2[0]:.3f}, チョキ={equilibrium_p2[1]:.3f}, パー={equilibrium_p2[2]:.3f}")
    
    # 理論的なナッシュ均衡（各手を1/3の確率で出す）との比較
    theoretical_equilibrium = np.array([1/3, 1/3, 1/3])
    error_p1 = np.linalg.norm(equilibrium_p1 - theoretical_equilibrium)
    error_p2 = np.linalg.norm(equilibrium_p2 - theoretical_equilibrium)
    print(f"\n理論値(1/3, 1/3, 1/3)との誤差: プレイヤー1={error_p1:.4f}, プレイヤー2={error_p2:.4f}")
    
    # ナッシュ均衡の検証
    print("\n3. ナッシュ均衡の検証")
    print("-" * 40)
    verification = rps.verify_nash_equilibrium(equilibrium_p1, equilibrium_p2)
    print(f"プレイヤー1の期待利得: {verification['p1_expected_payoff']:.4f}")
    print(f"プレイヤー1の後悔値: {verification['p1_regret']:.4f}")
    print(f"プレイヤー2の期待利得: {verification['p2_expected_payoff']:.4f}")
    print(f"プレイヤー2の後悔値: {verification['p2_regret']:.4f}")
    print("(後悔値が0に近いほど、ナッシュ均衡に近い)")
    
    # 収束過程の可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    history_array_p1 = np.array(history_p1)
    plt.plot(history_array_p1[:, 0], label='グー', color='red')
    plt.plot(history_array_p1[:, 1], label='チョキ', color='blue')
    plt.plot(history_array_p1[:, 2], label='パー', color='green')
    plt.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('反復回数 (×1000)')
    plt.ylabel('確率')
    plt.title('プレイヤー1の戦略の収束過程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    history_array_p2 = np.array(history_p2)
    plt.plot(history_array_p2[:, 0], label='グー', color='red')
    plt.plot(history_array_p2[:, 1], label='チョキ', color='blue')
    plt.plot(history_array_p2[:, 2], label='パー', color='green')
    plt.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('反復回数 (×1000)')
    plt.ylabel('確率')
    plt.title('プレイヤー2の戦略の収束過程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()