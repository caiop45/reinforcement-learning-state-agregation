import math
import time
import os
import csv
import traceback
import zlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import List, Tuple, Dict, Optional

from gridworld import GridworldEnv


def value_iteration(env, theta: float = 0.0001, discount_factor: float = 1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: ambiente tipo GridworldEnv. env.P representa as probabilidades
             de transição do ambiente, env.P[s][a] é uma lista de tuplas
             (prob, next_state, reward, done).
        theta: critério de parada; para quando a mudança máxima em V é < theta.
        discount_factor: fator de desconto gama.

    Returns:
        (policy, V): política ótima e função valor ótima.
    """

    def one_step_lookahead(state, V):
        """
        Calcula o valor esperado de cada ação em um estado.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.nS)
    iteration = 0
    while True:
        # Critério de parada
        delta = 0.0
        # Atualiza cada estado
        for s in range(env.nS):
            # One-step lookahead para achar a melhor ação
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calcula o delta
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Atualiza V(s)
            V[s] = best_action_value
        iteration += 1
        # Verifica se pode parar
        if delta < theta:
            print(f"Convergiu em {iteration} iterações.")
            break

    # Política determinística a partir de V*
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0

    return policy, V


def expected_value_function(env) -> np.ndarray:
    """
    Calcula analiticamente a função valor ótima para o gridworld
    com dois terminais nos cantos (0,0) e (h-1,w-1).
    """
    h, w = env.shape
    expected_v_grid = np.zeros(env.shape, dtype=float)
    for i in range(h):
        for j in range(w):
            dist1 = i + j  # distância até (0,0)
            dist2 = abs(h - 1 - i) + abs(w - 1 - j)  # distância até (h-1,w-1)
            expected_v_grid[i, j] = -min(dist1, dist2)
    return expected_v_grid.reshape(-1)


# === ALGORITMO 2: VALUE-BASED AGGREGATION ===

def value_based_aggregation(
    V: np.ndarray,
    epsilon: float
) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
    """
    Algoritmo 2: Value-based Aggregation.

    Args:
        V: Vetor de valores dos estados (shape: [nS])
        epsilon: Largura do intervalo para cada mega-estado

    Returns:
        mega_states: Lista de listas com indices dos estados por mega-estado
        W: Vetor de valores representativos (ponto medio de cada intervalo)
        state_to_mega: Vetor de tamanho nS com o índice do mega-estado de cada estado
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    V = np.asarray(V, dtype=float)
    nS = int(V.shape[0])

    b1 = float(np.min(V))
    b2 = float(np.max(V))
    if not (math.isfinite(b1) and math.isfinite(b2)):
        raise ValueError("V must contain only finite values")

    # Caso degenerado: todos os estados no mesmo valor.
    if b2 == b1:
        mega_states = [list(range(nS))]
        W = np.array([b1 + 0.5 * epsilon], dtype=float)
        state_to_mega = np.zeros(nS, dtype=np.int64)
        return mega_states, W, state_to_mega

    # Mantém a mesma definição de bins do código original:
    # bins i=0..num_bins-1 cobrem [b1+iε, b1+(i+1)ε) e o último inclui o valor b2.
    delta = (b2 - b1) / epsilon
    num_bins = max(1, int(math.ceil(delta)))

    raw_bin = np.floor((V - b1) / epsilon).astype(np.int64)
    raw_bin = np.clip(raw_bin, 0, num_bins - 1)

    # Remove bins vazios sem iterar sobre todos os bins possíveis.
    bin_ids, state_to_mega = np.unique(raw_bin, return_inverse=True)

    mega_states: List[List[int]] = [[] for _ in range(len(bin_ids))]
    for s, j in enumerate(state_to_mega.tolist()):
        mega_states[j].append(s)

    W = b1 + (bin_ids.astype(float) + 0.5) * epsilon
    return mega_states, W.astype(float), state_to_mega.astype(np.int64)


# === FUNCOES AUXILIARES ===

def expand_mega_state_values(
    mega_states: List[List[int]],
    W: np.ndarray,
    nS: int,
    state_to_mega: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Cria V_tilde expandindo valores dos mega-estados para estados individuais.
    """
    if state_to_mega is not None:
        state_to_mega = np.asarray(state_to_mega, dtype=np.int64)
        if int(state_to_mega.shape[0]) != int(nS):
            raise ValueError("state_to_mega must have length nS")
        if len(W) == 0:
            raise ValueError("W must be non-empty")
        if int(np.max(state_to_mega)) >= len(W) or int(np.min(state_to_mega)) < 0:
            raise ValueError("state_to_mega contains indices outside W")
        return np.asarray(W, dtype=float)[state_to_mega]

    if len(mega_states) != len(W):
        raise ValueError("mega_states and W must have the same length")

    V_tilde = np.zeros(nS, dtype=float)
    for idx, states in enumerate(mega_states):
        for s in states:
            if s < 0 or s >= nS:
                raise ValueError(f"state index {s} outside range [0, {nS})")
            V_tilde[s] = W[idx]
    return V_tilde


def bellman_operator_at_state(
    env,
    s: int,
    V_tilde: np.ndarray,
    discount_factor: float
) -> float:
    """
    Calcula Ts * V_tilde para um estado especifico.
    Ts V_tilde(s) = max_a sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V_tilde(s')]
    """
    action_values = []
    for a in range(env.nA):
        value = 0.0
        for prob, next_state, reward, done in env.P[s][a]:
            value += prob * (reward + discount_factor * V_tilde[next_state])
        action_values.append(value)
    return max(action_values)


def bellman_update_all_states(
    env,
    V: np.ndarray,
    discount_factor: float
) -> np.ndarray:
    """
    Aplica o operador Bellman em todos os estados.
    
    Args:
        env: ambiente tipo GridworldEnv
        V: vetor de valores dos estados (shape: [nS])
        discount_factor: fator de desconto gamma
        
    Returns:
        V_new: vetor atualizado com T(V) aplicado a todos os estados
    """
    V_new = np.zeros_like(V)
    for s in range(env.nS):
        V_new[s] = bellman_operator_at_state(env, s, V, discount_factor)
    return V_new


def run_value_iteration_until_convergence(
    env,
    discount_factor: float,
    theta: float = 0.0001,
) -> Tuple[np.ndarray, int, int, float]:
    """
    Executa Value Iteration até convergir e retorna métricas.

    Returns:
        V: função valor convergida
        iterations: número de iterações (varreduras completas)
        total_updates: número total de updates de estado (= iterations * nS)
        elapsed_s: tempo de execução em segundos
    """
    nS = env.nS
    V = np.zeros(nS, dtype=float)
    iterations = 0
    total_updates = 0

    start_time = time.perf_counter()
    while True:
        delta = 0.0
        for s in range(nS):
            old_value = V[s]
            V[s] = bellman_operator_at_state(env, s, V, discount_factor)
            delta = max(delta, abs(V[s] - old_value))
        iterations += 1
        total_updates += nS
        if delta < theta:
            break

    elapsed_s = time.perf_counter() - start_time
    return V, iterations, total_updates, elapsed_s


# === ALGORITMO 1: RANDOM VALUE ITERATION WITH AGGREGATION ===

def random_value_iteration_with_aggregation(
    env,
    mega_states: List[List[int]],
    W: np.ndarray,
    alpha: float,
    discount_factor: float,
    num_iterations: int = 1
) -> np.ndarray:
    """
    Algoritmo 1: Random Value Iteration with Aggregation.
    """
    if num_iterations < 1:
        return np.array(W, dtype=float, copy=True)

    W_new = np.array(W, dtype=float, copy=True)

    for _ in range(num_iterations):
        V_tilde = expand_mega_state_values(mega_states, W_new, env.nS)
        for j, states in enumerate(mega_states):
            if not states:
                continue
            sampled_state = int(np.random.choice(states))
            ts_value = bellman_operator_at_state(env, sampled_state, V_tilde, discount_factor)
            W_new[j] = (1.0 - alpha) * W_new[j] + alpha * ts_value

    return W_new


# === ALGORITMO 3: VALUE ITERATION WITH ADAPTIVE AGGREGATION ===

def value_iteration_with_adaptive_aggregation(
    env,
    epsilon: float = 0.5,
    discount_factor: float = 0.95,
    n_iterations: int = 1000,
    global_phase_size: int = 2,   # |B_i| - iterações globais por fase
    aggregate_phase_size: int = 5,  # |A_i| - iterações agregadas por fase
    V_star: np.ndarray = None,
    theoretical_bound: float = None,
    early_stop_on_bound: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Algoritmo 3: Value Iteration with Adaptive Aggregation.
    
    Alterna entre fases globais (Value Iteration padrão) e fases agregadas
    (atualizações em mega-estados com valores similares).
    
    Args:
        env: ambiente tipo GridworldEnv
        epsilon: largura dos bins de agregação
        discount_factor: fator de desconto gamma
        n_iterations: número total de iterações
        global_phase_size: |B_i| - iterações por fase global
        aggregate_phase_size: |A_i| - iterações por fase agregada
        
    Returns:
        V_final: vetor de valores final (shape: [nS])
        info: dicionário com métricas (iterações, atualizações, etc.)
    """
    nS = env.nS
    cycle_size = global_phase_size + aggregate_phase_size
    
    # Inicialização (linha 2 do Algorithm 3)
    V = np.zeros(nS, dtype=float)
    W = None
    mega_states = None
    state_to_mega = None
    t_sa = 1  # contador de iterações agregadas (para alpha_t)
    
    total_state_updates = 0  # métrica: número total de atualizações de estado
    converged_at = None
    early_stop_error = None
    
    for t in range(1, n_iterations + 1):
        # Determina posição no ciclo
        pos_in_cycle = (t - 1) % cycle_size
        
        # Fase global: B_i (posições 0 até global_phase_size-1)
        if pos_in_cycle < global_phase_size:
            # Início da fase global (linha 5-6)
            if pos_in_cycle == 0 and W is not None and mega_states is not None:
                # V_{t-1} = Ṽ(W_{t-1}) - expande valores dos mega-estados
                V = expand_mega_state_values(mega_states, W, nS, state_to_mega=state_to_mega)
            
            # Aplica Bellman em todos os estados (linha 7-8)
            V = bellman_update_all_states(env, V, discount_factor)
            total_state_updates += nS
            
        # Fase agregada: A_i (posições global_phase_size até cycle_size-1)
        else:
            # Início da fase agregada (linha 12-13)
            if pos_in_cycle == global_phase_size:
                # Chama Algoritmo 2 para criar agregação baseada em V atual
                mega_states, W, state_to_mega = value_based_aggregation(V, epsilon)
            
            # Calcula alpha_t = 1/sqrt(t_sa) (Seção 4.1.1)
            alpha_t = 1.0 / math.sqrt(t_sa)
            
            # Atualiza cada mega-estado (linhas 14-17)
            V_tilde = expand_mega_state_values(mega_states, W, nS, state_to_mega=state_to_mega)
            for j, states in enumerate(mega_states):
                if not states:
                    continue
                # Amostra estado uniformemente do mega-estado S_j
                sampled_state = int(np.random.choice(states))
                # Aplica atualização do Algoritmo 1 (linha 16)
                ts_value = bellman_operator_at_state(env, sampled_state, V_tilde, discount_factor)
                W[j] = (1.0 - alpha_t) * W[j] + alpha_t * ts_value
            
            total_state_updates += len(mega_states)  # K atualizações por iteração agregada
            t_sa += 1

        # Checa critério de parada baseado no limite teórico
        if early_stop_on_bound and V_star is not None and theoretical_bound is not None:
            V_current = V if pos_in_cycle < global_phase_size else expand_mega_state_values(
                mega_states, W, nS, state_to_mega=state_to_mega
            )
            current_error = np.max(np.abs(V_current - V_star))
            if current_error <= theoretical_bound:
                converged_at = t
                early_stop_error = current_error
                V_final = V_current
                info = {
                    'iterations': t,
                    'total_state_updates': total_state_updates,
                    't_sa_final': t_sa - 1,
                    'num_mega_states_final': len(mega_states) if mega_states else 0,
                    'stopped_early': True,
                    'early_stop_error': early_stop_error
                }
                return V_final, info
    
    # Retorno final (linhas 19-21)
    # Se terminou em fase global, retorna V_n
    pos_final = (n_iterations - 1) % cycle_size
    if pos_final < global_phase_size:
        V_final = V
    else:
        # Se terminou em fase agregada, retorna Ṽ(W_n)
        V_final = expand_mega_state_values(mega_states, W, nS, state_to_mega=state_to_mega)
    
    info = {
        'iterations': converged_at if converged_at is not None else n_iterations,
        'total_state_updates': total_state_updates,
        't_sa_final': t_sa - 1,  # número de iterações agregadas
        'num_mega_states_final': len(mega_states) if mega_states else 0,
        'stopped_early': converged_at is not None,
        'early_stop_error': early_stop_error
    }
    
    return V_final, info


# === COMPARAÇÃO DE ALGORITMOS ===

def compare_algorithms(
    env,
    V_star: np.ndarray = None,
    n_iterations: int = 1000,
    discount_factor: float = 0.95,
    epsilon: float = 0.5,
    global_phase_size: int = 2,
    aggregate_phase_size: int = 5
) -> Dict:
    """
    Compara Value Iteration padrão com o Algoritmo 3 (Adaptive Aggregation).
    
    Args:
        env: ambiente tipo GridworldEnv
        V_star: função valor ótima para comparação
        n_iterations: número de iterações para o Algoritmo 3
        discount_factor: fator de desconto gamma
        epsilon: largura dos bins de agregação
        global_phase_size: |B_i| - iterações globais por fase
        aggregate_phase_size: |A_i| - iterações agregadas por fase
        
    Returns:
        results: dicionário com métricas de ambos algoritmos
    """
    nS = env.nS
    
    # Se não foi fornecido V*, usa o próprio VI como referência (e evita rodar VI duas vezes no batch).
    compute_reference_via_vi = V_star is None

    # === Value Iteration Padrão ===
    print("\n" + "="*60)
    print("COMPARAÇÃO: Value Iteration vs Algoritmo 3")
    print("="*60)
    
    print("\n--- Value Iteration Padrão ---")

    theta = 0.0001  # critério de convergência
    V_vi, vi_iterations, vi_total_updates, vi_time = run_value_iteration_until_convergence(
        env, discount_factor=discount_factor, theta=theta
    )
    if compute_reference_via_vi:
        V_star = V_vi
        vi_error = 0.0
    else:
        vi_error = float(np.max(np.abs(V_vi - V_star)))  # norma infinito
    
    print(f"  Tempo de execução: {vi_time:.4f}s")
    print(f"  Iterações: {vi_iterations}")
    print(f"  Atualizações totais: {vi_total_updates}")
    print(f"  Erro ||V - V*||_∞: {vi_error:.6f}")
    
    theoretical_bound = 2 * epsilon / (1 - discount_factor)
    
    # === Algoritmo 3: Adaptive Aggregation ===
    print("\n--- Algoritmo 3: Adaptive Aggregation ---")
    start_time = time.perf_counter()
    
    V_algo3, algo3_info = value_iteration_with_adaptive_aggregation(
        env,
        epsilon=epsilon,
        discount_factor=discount_factor,
        n_iterations=n_iterations,
        global_phase_size=global_phase_size,
        aggregate_phase_size=aggregate_phase_size,
        V_star=V_star,
        theoretical_bound=theoretical_bound,
        early_stop_on_bound=True
    )
    
    algo3_time = time.perf_counter() - start_time
    algo3_error = np.max(np.abs(V_algo3 - V_star))  # norma infinito
    
    print(f"  Tempo de execução: {algo3_time:.4f}s")
    print(f"  Iterações: {algo3_info['iterations']}")
    print(f"  Atualizações totais: {algo3_info['total_state_updates']}")
    print(f"  Erro ||V - V*||_∞: {algo3_error:.6f}")
    print(f"  Mega-estados finais: {algo3_info['num_mega_states_final']}")
    print(f"  Iterações agregadas: {algo3_info['t_sa_final']}")
    
    # Limite teórico do erro (Teorema 1)
    print(f"\n  Limite teórico: 2ε/(1-γ) = {theoretical_bound:.4f}")
    print(f"  Erro dentro do limite: {'✓ Sim' if algo3_error <= theoretical_bound else '✗ Não'}")
    
    # === Resumo Comparativo ===
    print("\n--- Resumo Comparativo ---")
    speedup_updates = vi_total_updates / algo3_info['total_state_updates']
    print(f"  Redução de atualizações: {speedup_updates:.2f}x ({algo3_info['total_state_updates']} vs {vi_total_updates})")
    print(f"  Diferença de erro: {algo3_error - vi_error:.6f}")
    
    results = {
        'value_iteration': {
            'time': vi_time,
            'iterations': vi_iterations,
            'total_updates': vi_total_updates,
            'error': vi_error,
            'V': V_vi
        },
        'mega_state_agg': {
            'time': algo3_time,
            'iterations': algo3_info['iterations'],
            'total_updates': algo3_info['total_state_updates'],
            'error': algo3_error,
            'num_mega_states': algo3_info['num_mega_states_final'],
            't_sa_final': algo3_info['t_sa_final'],
            'stopped_early': algo3_info.get('stopped_early', False),
            'early_stop_error': algo3_info.get('early_stop_error', None),
            'V': V_algo3
        },
        'theoretical_bound': theoretical_bound,
        'speedup_updates': speedup_updates
    }
    
    return results


def _run_single_experiment(args):
    """
    Executa uma combinação de parâmetros e retorna duas linhas de resultado (VI e mega_state_agg).
    """
    (
        N,
        epsilon,
        transition_prob,
        discount_factor,
        global_phase_size,
        aggregate_phase_size,
        n_iterations,
    ) = args

    # Seed determinístico por configuração (evita correlação entre processos em execução paralela).
    seed_payload = f"{N}|{epsilon:.17g}|{transition_prob:.17g}|{discount_factor:.17g}|{global_phase_size}|{aggregate_phase_size}|{n_iterations}"
    seed = zlib.crc32(seed_payload.encode("utf-8")) & 0xFFFFFFFF
    np.random.seed(seed)

    # Cria ambiente
    env = GridworldEnv(shape=[N, N], transition_prob=transition_prob)

    # Compara algoritmos
    results = compare_algorithms(
        env,
        V_star=None,
        n_iterations=n_iterations,
        discount_factor=discount_factor,
        epsilon=epsilon,
        global_phase_size=global_phase_size,
        aggregate_phase_size=aggregate_phase_size,
    )

    vi_res = results['value_iteration']
    agg_res = results['mega_state_agg']

    # Retorna linhas prontas para o CSV
    return [
        {
            "N": N,
            "epsilon": epsilon,
            "transition_prob": transition_prob,
            "discount_factor": discount_factor,
            "global_phase_size": global_phase_size,
            "aggregate_phase_size": aggregate_phase_size,
            "algorithm": "value_iteration",
            "time": vi_res['time'],
            "iterations": vi_res['iterations'],
            "total_updates": vi_res['total_updates'],
            "error": vi_res['error'],
            "num_mega_states": "",
            "t_sa_final": "",
            "theoretical_bound": results['theoretical_bound'],
            "speedup_updates": results['speedup_updates'],
        },
        {
            "N": N,
            "epsilon": epsilon,
            "transition_prob": transition_prob,
            "discount_factor": discount_factor,
            "global_phase_size": global_phase_size,
            "aggregate_phase_size": aggregate_phase_size,
            "algorithm": "mega_state_agg",
            "time": agg_res['time'],
            "iterations": agg_res['iterations'],
            "total_updates": agg_res['total_updates'],
            "error": agg_res['error'],
            "num_mega_states": agg_res['num_mega_states'],
            "t_sa_final": agg_res.get('t_sa_final', ''),
            "theoretical_bound": results['theoretical_bound'],
            "speedup_updates": results['speedup_updates'],
        },
    ]


def run_all_experiments(
    grid_sizes: List[int],
    epsilons: List[float],
    transition_probs: List[float],
    discount_factors: List[float],
    global_phase_size: int = 2,
    aggregate_phase_size: int = 5,
    n_iterations: int = 1000,
    csv_filename: str = "experiment_results.csv",
    failures_filename: str = "experiment_failures.csv",
    resume: bool = False,
    max_workers: Optional[int] = None,
    phase_pairs: Optional[List[Tuple[int, int]]] = None,
):
    """
    Executa experimentos para todas as combinações de parâmetros.
    
    Args:
        grid_sizes: lista de tamanhos de grid (N x N)
        epsilons: lista de valores de epsilon para agregação
        transition_probs: lista de probabilidades de transição
        discount_factors: lista de fatores de desconto
        global_phase_size: |B_i| - iterações globais por fase
        aggregate_phase_size: |A_i| - iterações agregadas por fase
        n_iterations: número de iterações para o Algoritmo 3
        csv_filename: nome do arquivo CSV para salvar resultados
        phase_pairs: lista de pares (global_phase_size, aggregate_phase_size) para varrer no mesmo batch
    """
    # Caminho do arquivo CSV para salvar resultados
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_filename)
    failures_path = os.path.join(script_dir, failures_filename)

    fieldnames = [
        "N",
        "epsilon",
        "transition_prob",
        "discount_factor",
        "global_phase_size",
        "aggregate_phase_size",
        "algorithm",
        "time",
        "iterations",
        "total_updates",
        "error",
        "num_mega_states",
        "t_sa_final",
        "theoretical_bound",
        "speedup_updates",
    ]

    config_cols = [
        "N",
        "epsilon",
        "transition_prob",
        "discount_factor",
        "global_phase_size",
        "aggregate_phase_size",
    ]

    def canonical_config_key(
        N: int,
        epsilon: float,
        transition_prob: float,
        discount_factor: float,
        global_phase_size: int,
        aggregate_phase_size: int,
    ) -> Tuple[int, float, float, float, int, int]:
        return (
            int(N),
            float(epsilon),
            float(transition_prob),
            float(discount_factor),
            int(global_phase_size),
            int(aggregate_phase_size),
        )

    def normalize_algorithm_name(name: str) -> str:
        if name == "algorithm_3":
            return "mega_state_agg"
        return name

    def load_completed_configs(path: str) -> set:
        if not os.path.exists(path):
            return set()
        completed = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return set()
            for row in reader:
                try:
                    key = canonical_config_key(
                        int(float(row.get("N", 0))),
                        float(row.get("epsilon", 0.0)),
                        float(row.get("transition_prob", 0.0)),
                        float(row.get("discount_factor", 0.0)),
                        int(float(row.get("global_phase_size", 0))),
                        int(float(row.get("aggregate_phase_size", 0))),
                    )
                except Exception:
                    continue
                alg = normalize_algorithm_name(row.get("algorithm", ""))
                completed.setdefault(key, set()).add(alg)
        return {k for k, algs in completed.items() if {"value_iteration", "mega_state_agg"}.issubset(algs)}

    # Cria/limpa o arquivo CSV (ou retoma se solicitado).
    if (not resume) or (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0):
        with open(csv_path, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    completed_configs = load_completed_configs(csv_path) if resume else set()

    # Calcula o total de combinações
    if phase_pairs is None:
        phase_pairs = [(int(global_phase_size), int(aggregate_phase_size))]
    phase_pairs = [(int(b), int(a)) for (b, a) in phase_pairs]
    total_combinations = (
        len(grid_sizes)
        * len(epsilons)
        * len(transition_probs)
        * len(discount_factors)
        * len(phase_pairs)
    )
    
    print(f"\n{'='*70}")
    print(f"INICIANDO EXPERIMENTOS - {total_combinations} combinações no total")
    print(f"{'='*70}")
    print(f"Grid sizes: {grid_sizes}")
    print(f"Epsilons: {epsilons}")
    print(f"Transition probs: {transition_probs}")
    print(f"Discount factors: {discount_factors}")
    print(f"Phase pairs (|B|,|A|): {phase_pairs}")
    print(f"{'='*70}\n")

    # Falhas são registradas sem interromper o batch.
    failures_fieldnames = config_cols + ["exception_type", "exception_message", "traceback"]
    if (not resume) or (not os.path.exists(failures_path)) or (os.path.getsize(failures_path) == 0):
        with open(failures_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=failures_fieldnames)
            writer.writeheader()

    def default_max_workers_for_grid_size(N: int) -> int:
        # Ambientes grandes consomem muita memória (env.P é grande); limitar paralelismo evita interrupções.
        cpu = os.cpu_count() or 1
        if N >= 400:
            return 1
        if N >= 200:
            return max(1, min(2, cpu))
        return max(1, min(4, cpu))

    # Executa por N para poder ajustar max_workers de forma segura em grids grandes.
    total_tasks = total_combinations
    completed_count = 0
    skipped_count = 0
    failed_count = 0

    with open(csv_path, mode="a", newline="") as csvfile, open(failures_path, mode="a", newline="") as failfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        fail_writer = csv.DictWriter(failfile, fieldnames=failures_fieldnames)

        for N in grid_sizes:
            tasks = []
            for epsilon, transition_prob, discount_factor in product(epsilons, transition_probs, discount_factors):
                for (b_size, a_size) in phase_pairs:
                    key = canonical_config_key(
                        int(N),
                        float(epsilon),
                        float(transition_prob),
                        float(discount_factor),
                        int(b_size),
                        int(a_size),
                    )
                    if resume and key in completed_configs:
                        skipped_count += 1
                        continue
                    tasks.append((N, epsilon, transition_prob, discount_factor, b_size, a_size, n_iterations))

            if not tasks:
                continue

            workers = max_workers if max_workers is not None else default_max_workers_for_grid_size(int(N))
            print(f"[N={N}] Executando {len(tasks)} tarefas com max_workers={workers}...")

            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_args = {executor.submit(_run_single_experiment, t): t for t in tasks}
                for future in as_completed(future_to_args):
                    t = future_to_args[future]
                    try:
                        rows = future.result()
                        for row in rows:
                            row["algorithm"] = normalize_algorithm_name(row.get("algorithm", ""))
                            writer.writerow(row)
                        csvfile.flush()
                        completed_count += 1
                    except Exception as exc:
                        failed_count += 1
                        N0, eps0, p0, g0, B0, A0, _ = t
                        fail_writer.writerow(
                            {
                                "N": N0,
                                "epsilon": eps0,
                                "transition_prob": p0,
                                "discount_factor": g0,
                                "global_phase_size": B0,
                                "aggregate_phase_size": A0,
                                "exception_type": type(exc).__name__,
                                "exception_message": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        failfile.flush()
                        print(f"[FALHA] N={N0} eps={eps0} p={p0} gamma={g0}: {type(exc).__name__}: {exc}")

    print(f"\n{'='*70}")
    print(f"EXPERIMENTOS CONCLUÍDOS!")
    print(f"Resultados salvos em: {csv_path}")
    print(f"Falhas (se houver) em: {failures_path}")
    print(f"Total de combinações testadas: {total_combinations}")
    print(f"{'='*70}\n")

    return {
        "total_combinations": total_tasks,
        "completed_configs_written": completed_count,
        "skipped_configs": skipped_count,
        "failed_configs": failed_count,
        "csv_path": csv_path,
        "failures_path": failures_path,
    }


if __name__ == "__main__":
    # Parâmetros a serem testados
    grid_sizes = [10, 20, 50, 100, 200, 400, 500]
    epsilons = [0.0005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5]
    transition_probs = [0.5, 0.6, 0.7, 0.92, 0.95, 0.98, 1.0]
    discount_factors = [0.8, 0.85, 0.90, 0.95, 0.99]
    phase_pairs = [(2, 5), (2, 10), (5, 20)]
    
    # Executa todos os experimentos
    run_all_experiments(
        grid_sizes=grid_sizes,
        epsilons=epsilons,
        transition_probs=transition_probs,
        discount_factors=discount_factors,
        n_iterations=1000,
        csv_filename="experiment_results.csv",
        phase_pairs=phase_pairs,
    )
