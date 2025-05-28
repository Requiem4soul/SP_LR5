import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


# =============================================================================
# ЗАДАЧА 1: Волатильность функции log(S(t))
# =============================================================================

def task1_volatility_log_price():
    """
    Задача 1: Найти волатильность функции log(S(t))

    Дано: dS(t) = μS(t)dt + σS(t)dW(t) - геометрическое броуновское движение
    Найти: волатильность log(S(t))

    Решение через формулу Ито:
    Для f(S) = log(S):
    f'(S) = 1/S
    f''(S) = -1/S²

    По формуле Ито:
    df = f'(S)dS + (1/2)f''(S)(dS)²
    df = (1/S)[μS dt + σS dW] + (1/2)(-1/S²)(σS)²dt
    df = μ dt + σ dW - (1/2)σ² dt
    df = (μ - σ²/2) dt + σ dW

    Волатильность log(S(t)) = σ (коэффициент при dW)
    """
    print("ЗАДАЧА 1: Волатильность функции log(S(t))")
    print("=" * 50)

    # Параметры модели
    mu = 0.05  # средняя доходность (5%)
    sigma = 0.2  # волатильность цены акции (20%)

    print(f"Исходные параметры:")
    print(f"μ (средняя доходность) = {mu}")
    print(f"σ (волатильность цены) = {sigma}")

    # Применяем формулу Ито для log(S)
    volatility_log_S = sigma
    drift_log_S = mu - 0.5 * sigma ** 2

    print(f"\nРезультат применения формулы Ито:")
    print(f"d[log(S)] = ({drift_log_S:.4f})dt + ({volatility_log_S})dW")
    print(f"\nВолатильность log(S(t)) = {volatility_log_S}")

    # Численная проверка через симуляцию
    print("\nЧисленная проверка:")
    T = 1.0
    N = 1000
    dt = T / N
    S0 = 100

    # Симуляция геометрического броуновского движения
    np.random.seed(42)
    dW = np.random.normal(0, np.sqrt(dt), N)

    S = np.zeros(N + 1)
    log_S = np.zeros(N + 1)
    S[0] = S0
    log_S[0] = np.log(S0)

    for i in range(N):
        # Точная формула для геом. броуновского движения
        S[i + 1] = S[i] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW[i])
        log_S[i + 1] = np.log(S[i + 1])

    # Вычисляем эмпирическую волатильность log(S)
    log_returns = np.diff(log_S)
    empirical_volatility = np.std(log_returns) / np.sqrt(dt)

    print(f"Теоретическая волатильность log(S): {volatility_log_S}")
    print(f"Эмпирическая волатильность log(S): {empirical_volatility:.4f}")

    return volatility_log_S


# =============================================================================
# ЗАДАЧА 2: Среднее изменение популяции
# =============================================================================

def task2_population_growth():
    """
    Задача 2: Найти среднее изменение популяции за единицу времени

    Дано: dX(t) = rX(t)dt + σX(t)dW(t)
    Найти: E[dX(t)/dt] - среднее изменение за единицу времени

    Решение:
    E[dX(t)] = E[rX(t)dt + σX(t)dW(t)]
    E[dX(t)] = rE[X(t)]dt + σE[X(t)]E[dW(t)]
    Поскольку E[dW(t)] = 0:
    E[dX(t)] = rE[X(t)]dt

    Среднее изменение за единицу времени: E[dX(t)/dt] = rE[X(t)]
    """
    print("\n\nЗАДАЧА 2: Среднее изменение популяции")
    print("=" * 50)

    # Параметры модели
    r = 0.03  # скорость роста популяции
    sigma = 0.1  # волатильность роста
    X0 = 1000  # начальная популяция

    print(f"Параметры модели:")
    print(f"r (скорость роста) = {r}")
    print(f"σ (волатильность) = {sigma}")
    print(f"X(0) = {X0}")

    # Теоретическое решение
    # Для геометрического броуновского движения: E[X(t)] = X(0)e^(rt)
    def expected_population(t):
        return X0 * np.exp(r * t)

    def mean_change_rate(t):
        return r * expected_population(t)

    print(f"\nТеоретическое решение:")
    print(f"E[X(t)] = X(0)·e^(rt) = {X0}·e^({r}t)")
    print(f"Среднее изменение за единицу времени: E[dX(t)/dt] = r·E[X(t)]")

    # Примеры для разных моментов времени
    times = [0, 1, 2, 5, 10]
    print(f"\nПримеры для разных моментов времени:")
    for t in times:
        exp_pop = expected_population(t)
        mean_change = mean_change_rate(t)
        print(f"t = {t:2d}: E[X(t)] = {exp_pop:8.1f}, E[dX/dt] = {mean_change:6.2f}")

    # Численная проверка
    print(f"\nЧисленная проверка (симуляция):")
    T = 5.0
    N = 1000
    dt = T / N
    num_simulations = 1000

    np.random.seed(42)
    final_populations = []

    for _ in range(num_simulations):
        X = X0
        for i in range(N):
            dW = np.random.normal(0, np.sqrt(dt))
            X = X * (1 + r * dt + sigma * dW)
        final_populations.append(X)

    empirical_mean = np.mean(final_populations)
    theoretical_mean = expected_population(T)

    print(f"После {T} лет:")
    print(f"Теоретическое E[X({T})] = {theoretical_mean:.1f}")
    print(f"Эмпирическое E[X({T})] = {empirical_mean:.1f}")

    return mean_change_rate


# =============================================================================
# ЗАДАЧА 3: Применение формулы Ито
# =============================================================================

def task3_ito_formula():
    """
    Задача 3: Применить формулу Ито для S(t,W(t)) = S(0)·exp((μ-0.5σ²)t + σW(t))

    Найти: dS(t) используя формулу Ито

    Решение:
    Пусть f(t,W) = S(0)·exp((μ-0.5σ²)t + σW)

    Частные производные:
    ∂f/∂t = S(0)·(μ-0.5σ²)·exp((μ-0.5σ²)t + σW) = (μ-0.5σ²)·f
    ∂f/∂W = S(0)·σ·exp((μ-0.5σ²)t + σW) = σ·f
    ∂²f/∂W² = S(0)·σ²·exp((μ-0.5σ²)t + σW) = σ²·f

    По формуле Ито:
    df = (∂f/∂t + (1/2)∂²f/∂W²)dt + (∂f/∂W)dW
    df = ((μ-0.5σ²)f + 0.5σ²f)dt + σf dW
    df = μf dt + σf dW

    Поэтому: dS = μS dt + σS dW
    """
    print("\n\nЗАДАЧА 3: Применение формулы Ито")
    print("=" * 50)

    # Параметры
    mu = 0.08
    sigma = 0.25
    S0 = 100

    print(f"Параметры:")
    print(f"S(0) = {S0}")
    print(f"μ = {mu}")
    print(f"σ = {sigma}")

    print(f"\nИсходная функция:")
    print(f"S(t,W(t)) = S(0)·exp((μ-0.5σ²)t + σW(t))")
    print(f"S(t,W(t)) = {S0}·exp(({mu}-0.5·{sigma}²)t + {sigma}W(t))")
    print(f"S(t,W(t)) = {S0}·exp({mu - 0.5 * sigma ** 2:.4f}t + {sigma}W(t))")

    print(f"\nВычисление частных производных:")
    print(f"∂S/∂t = (μ-0.5σ²)·S = {mu - 0.5 * sigma ** 2:.4f}·S")
    print(f"∂S/∂W = σ·S = {sigma}·S")
    print(f"∂²S/∂W² = σ²·S = {sigma ** 2}·S")

    print(f"\nПрименение формулы Ито:")
    print(f"dS = (∂S/∂t + 0.5·∂²S/∂W²)dt + (∂S/∂W)dW")
    print(f"dS = ({mu - 0.5 * sigma ** 2:.4f}·S + 0.5·{sigma ** 2}·S)dt + {sigma}·S·dW")
    print(f"dS = ({mu - 0.5 * sigma ** 2 + 0.5 * sigma ** 2:.4f}·S)dt + {sigma}·S·dW")
    print(f"dS = {mu}·S·dt + {sigma}·S·dW")

    print(f"\nИтоговый результат:")
    print(f"dS(t) = μS(t)dt + σS(t)dW(t)")
    print(f"Это стандартное геометрическое броуновское движение!")

    # Численная проверка
    print(f"\nЧисленная проверка:")
    T = 1.0
    N = 1000
    dt = T / N

    np.random.seed(42)

    # Метод 1: Прямое использование формулы
    W = np.cumsum(np.random.normal(0, np.sqrt(dt), N))
    t = np.arange(N + 1) * dt
    W = np.concatenate([[0], W])

    S_formula = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)

    # Метод 2: Численное интегрирование SDE
    S_sde = np.zeros(N + 1)
    S_sde[0] = S0

    for i in range(N):
        dW = W[i + 1] - W[i]
        S_sde[i + 1] = S_sde[i] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW)

    print(f"Конечное значение (формула): S({T}) = {S_formula[-1]:.2f}")
    print(f"Конечное значение (SDE): S({T}) = {S_sde[-1]:.2f}")
    print(f"Относительная разность: {abs(S_formula[-1] - S_sde[-1]) / S_formula[-1] * 100:.4f}%")

    return S_formula, S_sde, t


# =============================================================================
# ЗАДАЧА 4: Математическое ожидание инфляции
# =============================================================================

def task4_inflation_expectation():
    """
    Задача 4: Инфляция подчиняется dI(t) = μI(t)dt + σI(t)dW(t)

    Найти: 
    1) Математическое ожидание инфляции в будущем
    2) Среднее изменение логарифма инфляции

    Решение:
    1) Для геометрического броуновского движения: E[I(t)] = I(0)e^(μt)
    2) Для log(I): d[log(I)] = (μ - σ²/2)dt + σdW
       Среднее изменение log(I) за единицу времени = μ - σ²/2
    """
    print("\n\nЗАДАЧА 4: Математическое ожидание инфляции")
    print("=" * 50)

    # Параметры
    mu = 0.02  # средний темп роста инфляции (2%)
    sigma = 0.05  # волатильность инфляции (5%)
    I0 = 0.03  # начальная инфляция (3%)

    print(f"Параметры модели инфляции:")
    print(f"μ (средний темп роста) = {mu}")
    print(f"σ (волатильность) = {sigma}")
    print(f"I(0) (начальная инфляция) = {I0}")

    # 1. Математическое ожидание инфляции
    def expected_inflation(t):
        return I0 * np.exp(mu * t)

    print(f"\n1. Математическое ожидание инфляции:")
    print(f"E[I(t)] = I(0)·e^(μt) = {I0}·e^({mu}t)")

    # Примеры для разных периодов
    periods = [1, 2, 5, 10, 20]
    print(f"\nПрогноз инфляции на различные периоды:")
    for t in periods:
        exp_inflation = expected_inflation(t)
        print(f"Через {t:2d} лет: E[I({t})] = {exp_inflation:.4f} ({exp_inflation * 100:.2f}%)")

    # 2. Среднее изменение логарифма инфляции
    log_drift = mu - 0.5 * sigma ** 2
    log_volatility = sigma

    print(f"\n2. Среднее изменение логарифма инфляции:")
    print(f"d[log(I)] = (μ - σ²/2)dt + σdW")
    print(f"d[log(I)] = ({mu} - {sigma}²/2)dt + {sigma}dW")
    print(f"d[log(I)] = {log_drift:.6f}dt + {sigma}dW")
    print(f"\nСреднее изменение log(I) за единицу времени = {log_drift:.6f}")

    # Численная проверка
    print(f"\nЧисленная проверка (симуляция):")
    T = 10.0
    N = 1000
    dt = T / N
    num_simulations = 10000

    np.random.seed(42)
    final_inflations = []
    log_changes = []

    for _ in range(num_simulations):
        I = I0
        log_I_initial = np.log(I0)

        for i in range(N):
            dW = np.random.normal(0, np.sqrt(dt))
            I = I * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW)

        final_inflations.append(I)
        log_I_final = np.log(I)
        log_changes.append((log_I_final - log_I_initial) / T)

    empirical_mean_inflation = np.mean(final_inflations)
    theoretical_mean_inflation = expected_inflation(T)

    empirical_log_change = np.mean(log_changes)
    theoretical_log_change = log_drift

    print(f"После {T} лет:")
    print(f"Теоретическое E[I({T})] = {theoretical_mean_inflation:.6f}")
    print(f"Эмпирическое E[I({T})] = {empirical_mean_inflation:.6f}")
    print(f"\nИзменение логарифма за единицу времени:")
    print(f"Теоретическое = {theoretical_log_change:.6f}")
    print(f"Эмпирическое = {empirical_log_change:.6f}")

    return expected_inflation, log_drift


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    """Запуск всех задач лабораторной работы"""
    print("ЛАБОРАТОРНАЯ РАБОТА ПО СТОХАСТИЧЕСКОМУ АНАЛИЗУ")
    print("=" * 60)

    # Выполняем все задачи
    task1_result = task1_volatility_log_price()
    task2_result = task2_population_growth()
    task3_result = task3_ito_formula()
    task4_result = task4_inflation_expectation()

    print("\n" + "=" * 60)
    print("КРАТКОЕ РЕЗЮМЕ РЕЗУЛЬТАТОВ:")
    print("=" * 60)
    print(f"Задача 1: Волатильность log(S(t)) = исходная волатильность σ")
    print(f"Задача 2: Среднее изменение популяции = r·E[X(t)]")
    print(f"Задача 3: Формула Ито подтверждает геометрическое броуновское движение")
    print(f"Задача 4: E[I(t)] = I(0)e^(μt), среднее изменение log(I) = μ - σ²/2")


if __name__ == "__main__":
    main()