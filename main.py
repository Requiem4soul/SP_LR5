import numpy as np
import matplotlib.pyplot as plt


def task1_volatility_log_price():
    print("Задача 1: Волатильность функции log(S(t))")

    mu = 0.05
    sigma = 0.2
    print(f"Исходные параметры:\nсредняя доходность = {mu}\nволатильность цены = {sigma}")

    volatility_log_S = sigma
    drift_log_S = mu - 0.5 * sigma ** 2

    print(f"\nРезультат применения формулы Ито:\nd[log(S)] = ({drift_log_S:.4f})dt + ({volatility_log_S})dW")
    print(f"Волатильность log(S(t)) = {volatility_log_S}")

    T = 1.0
    N = 1000
    dt = T / N
    S0 = 100

    time = np.linspace(0, T, N + 1)
    expected_log_S = np.log(S0) + drift_log_S * time
    upper_bound = expected_log_S + 3 * sigma * np.sqrt(time)
    lower_bound = expected_log_S - 3 * sigma * np.sqrt(time)

    num_paths = 10
    all_log_S_paths = []
    all_log_returns = []

    np.random.seed(42)
    for _ in range(num_paths):
        dW = np.random.normal(0, np.sqrt(dt), N)
        S = np.zeros(N + 1)
        log_S = np.zeros(N + 1)
        S[0] = S0
        log_S[0] = np.log(S0)
        for i in range(N):
            S[i + 1] = S[i] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW[i])
            log_S[i + 1] = np.log(S[i + 1])
        all_log_S_paths.append(log_S)
        all_log_returns.extend(np.diff(log_S))

    empirical_volatility = np.std(all_log_returns) / np.sqrt(dt)
    print(f"Теоретическая волатильность log(S): {volatility_log_S}")
    print(f"Эмпирическая волатильность log(S): {empirical_volatility:.4f}")

    plt.figure(figsize=(10, 6))
    for log_S in all_log_S_paths:
        plt.plot(time, log_S, color="blue", alpha=0.4)

    plt.plot(time, expected_log_S, label="Теоретическое ожидание log(S(t))", color="orange")
    plt.plot(time, upper_bound, linestyle="--", color="gray", label="±3σ√t")
    plt.plot(time, lower_bound, linestyle="--", color="gray")

    plt.title("Задача 1: log(S(t)) и доверительная область")
    plt.xlabel("t")
    plt.ylabel("log(S(t))")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def task2_population_growth():
    print("\nЗадача 2: Среднее изменение популяции")

    r = 0.03
    sigma = 0.1
    X0 = 1000

    def expected_population(t):
        return X0 * np.exp(r * t)

    def expected_logX(t):
        return np.log(X0) + (r - 0.5 * sigma ** 2) * t

    T = 10.0
    N = 1000
    dt = T / N
    time = np.linspace(0, T, N + 1)

    num_paths = 1000
    X_all = np.zeros((num_paths, N + 1))
    np.random.seed(42)
    for j in range(num_paths):
        X = np.zeros(N + 1)
        X[0] = X0
        for i in range(N):
            dW = np.random.normal(0, np.sqrt(dt))
            X[i + 1] = X[i] * (1 + r * dt + sigma * dW)
        X_all[j] = X

    empirical_mean = X_all[:, -1].mean()
    theoretical_mean = expected_population(T)
    print(f"После {T} лет:")
    print(f"Теоретическое E[X({T})] = {theoretical_mean:.1f}")
    print(f"Эмпирическое E[X({T})] = {empirical_mean:.1f}")

    # Логарифмическое изменение
    log_X_all = np.log(X_all)
    empirical_log_change = (log_X_all[:, -1] - log_X_all[:, 0]).mean() / T
    theoretical_log_change = r - 0.5 * sigma ** 2

    print(f"\nИзменение логарифма за единицу времени:")
    print(f"Теоретическое = {theoretical_log_change:.6f}")
    print(f"Эмпирическое = {empirical_log_change:.6f}")

    # Визуализация с логнормальной зоной
    num_plot_paths = 10
    X_plot_paths = X_all[:num_plot_paths]

    expected = expected_population(time)
    upper = np.exp(expected_logX(time) + 3 * sigma * np.sqrt(time))
    lower = np.exp(expected_logX(time) - 3 * sigma * np.sqrt(time))

    plt.figure(figsize=(10, 6))
    for X in X_plot_paths:
        plt.plot(time, X, color="blue", alpha=0.3)

    plt.plot(time, expected, color="orange", label="Теоретическое ожидание E[X(t)]")
    plt.plot(time, upper, linestyle="--", color="gray", label="±3σ логнормальная область")
    plt.plot(time, lower, linestyle="--", color="gray")
    plt.title("Задача 2: X(t) и логнормальная доверительная зона")
    plt.xlabel("t")
    plt.ylabel("X(t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def task3_ito_formula():
    print("\nЗадача 3: Применение формулы Ито")

    mu = 0.08
    sigma = 0.25
    S0 = 100

    T = 1.0
    N = 1000
    dt = T / N
    t = np.linspace(0, T, N + 1)
    np.random.seed(42)

    W = np.cumsum(np.random.normal(0, np.sqrt(dt), N))
    W = np.concatenate([[0], W])

    S_formula = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)

    S_sde = np.zeros(N + 1)
    S_sde[0] = S0
    for i in range(N):
        dW = W[i + 1] - W[i]
        S_sde[i + 1] = S_sde[i] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW)

    print(f"Значение с применением формулы Ито: S({T}) = {S_formula[-1]:.2f}")
    print(f"Реальное значение для этого же случая: S({T}) = {S_sde[-1]:.2f}")
    rel_diff = abs(S_formula[-1] - S_sde[-1]) / S_formula[-1] * 100
    print(f"Относительная разность: {rel_diff:.4f}%")

    plt.figure(figsize=(8, 5))
    plt.plot(t, S_formula, label="S по формуле")
    plt.plot(t, S_sde, linestyle="--", label="S по SDE")
    plt.title("Задача 3: Сравнение S(t) по двум подходам")
    plt.xlabel("t")
    plt.ylabel("S(t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def task4_inflation_expectation():
    print("\nЗадача 4: Математическое ожидание инфляции")

    mu = 0.02
    sigma = 0.05
    I0 = 0.03

    def expected_inflation(t):
        return I0 * np.exp(mu * t)

    def expected_logI(t):
        return np.log(I0) + (mu - 0.5 * sigma ** 2) * t

    T = 20.0
    N = 1000
    dt = T / N
    t = np.linspace(0, T, N + 1)
    num_paths = 1000

    I_all = np.zeros((num_paths, N + 1))
    log_I_all = np.zeros((num_paths, N + 1))

    np.random.seed(42)
    for j in range(num_paths):
        I = np.zeros(N + 1)
        log_I = np.zeros(N + 1)
        I[0] = I0
        log_I[0] = np.log(I0)
        for i in range(N):
            dW = np.random.normal(0, np.sqrt(dt))
            I[i + 1] = I[i] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW)
            log_I[i + 1] = np.log(I[i + 1])
        I_all[j] = I
        log_I_all[j] = log_I

    empirical_mean_inflation = I_all[:, -1].mean()
    theoretical_mean_inflation = expected_inflation(T)
    empirical_log_change = (log_I_all[:, -1] - log_I_all[:, 0]).mean() / T
    theoretical_log_change = mu - 0.5 * sigma ** 2

    print(f"После {T} лет:")
    print(f"Теоретическое E[I({T})] = {theoretical_mean_inflation:.6f}")
    print(f"Эмпирическое E[I({T})] = {empirical_mean_inflation:.6f}")
    print(f"\nИзменение логарифма за единицу времени:")
    print(f"Теоретическое = {theoretical_log_change:.6f}")
    print(f"Эмпирическое = {empirical_log_change:.6f}")

    num_plot = 10
    I_plot_paths = I_all[:num_plot]

    expected = expected_inflation(t)
    upper = np.exp(expected_logI(t) + 3 * sigma * np.sqrt(t))
    lower = np.exp(expected_logI(t) - 3 * sigma * np.sqrt(t))

    plt.figure(figsize=(10, 6))
    for I in I_plot_paths:
        plt.plot(t, I, color="blue", alpha=0.4)

    plt.plot(t, expected, color="orange", label="Теоретическое ожидание E[I(t)]")
    plt.plot(t, upper, linestyle="--", color="gray", label="±3σ логнормальная область")
    plt.plot(t, lower, linestyle="--", color="gray")
    plt.title("Задача 4: I(t) и логнормальная доверительная зона")
    plt.xlabel("t")
    plt.ylabel("I(t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    task1_volatility_log_price()
    task2_population_growth()
    task3_ito_formula()
    task4_inflation_expectation()


if __name__ == "__main__":
    main()
