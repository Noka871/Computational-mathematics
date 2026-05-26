import numpy as np
from matrix_utils import print_system, check_convergence, residual
from gauss import gauss_solve
from iteration import simple_iteration, seidel_method
from test_systems import get_system


def main():
    A, b = get_system()

    print("=" * 50)
    print_system(A, b)
    print(f"\nТочность: ε = 0.01")

    if check_convergence(A):
        print("Условие сходимости: выполняется")
    else:
        print("Условие сходимости: не выполняется")

    print("\n" + "=" * 50)
    print("1. МЕТОД ГАУССА")
    print("=" * 50)

    x_gauss = gauss_solve(A.copy(), b.copy())
    res_gauss = residual(A, x_gauss, b)
    print(f"Решение: {x_gauss}")
    print(f"Невязка: {res_gauss:.2e}")

    print("\n" + "=" * 50)
    print("2. МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
    print("=" * 50)

    x_iter, iter_iter = simple_iteration(A, b)
    res_iter = residual(A, x_iter, b)
    print(f"Решение: {x_iter}")
    print(f"Невязка: {res_iter:.2e}")
    print(f"Итераций: {iter_iter}")

    print("\n" + "=" * 50)
    print("3. МЕТОД ЗЕЙДЕЛЯ")
    print("=" * 50)

    x_seidel, iter_seidel = seidel_method(A, b)
    res_seidel = residual(A, x_seidel, b)
    print(f"Решение: {x_seidel}")
    print(f"Невязка: {res_seidel:.2e}")
    print(f"Итераций: {iter_seidel}")

    print("\n" + "=" * 50)
    print("СРАВНЕНИЕ МЕТОДОВ")
    print("=" * 50)

    print(f"{'Метод':<20} {'x1':<10} {'x2':<10} {'x3':<10} {'Невязка':<12} {'Итер.'}")
    print("-" * 65)
    print(f"{'Гаусс':<20} {x_gauss[0]:<10.6f} {x_gauss[1]:<10.6f} {x_gauss[2]:<10.6f} {res_gauss:<12.2e} 1")
    print(
        f"{'Простая итерация':<20} {x_iter[0]:<10.6f} {x_iter[1]:<10.6f} {x_iter[2]:<10.6f} {res_iter:<12.2e} {iter_iter}")
    print(
        f"{'Зейдель':<20} {x_seidel[0]:<10.6f} {x_seidel[1]:<10.6f} {x_seidel[2]:<10.6f} {res_seidel:<12.2e} {iter_seidel}")

    print("\n" + "=" * 50)
    print("АНАЛИЗ")
    print("=" * 50)

    print("1. Метод Гаусса дал наиболее точное решение.")
    print(f"2. Метод Зейделя сходится за {iter_seidel} итераций,")
    print(f"   метод простой итерации - за {iter_iter} итераций.")

    if iter_seidel < iter_iter:
        print(f"3. Метод Зейделя сходится быстрее в {iter_iter / iter_seidel:.1f} раз.")
    else:
        print(f"3. Метод простой итерации сходится быстрее в {iter_seidel / iter_iter:.1f} раз.")


if __name__ == "__main__":
    main()
