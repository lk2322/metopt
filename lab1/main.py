"""
решение задачи линейного программирования
двухфазным симплекс-методом.

Формат входного файла (пример):
max 2 3 1 4
1 1 1 1 <= 10
2 1 -1 1 = 8
0 1 2 1 >= 5

Запуск:
    python main problem.txt

Вывод: в консоль и в файл result.txt
"""

import sys
import copy
from typing import List, Optional, Tuple


# ===============================
# Основной класс решателя
# ===============================
class TwoPhaseSimplex:
    def __init__(self, num_vars: int, A: List[List[float]], signs: List[str],
                 b: List[float], c: List[float], sense: str = 'max'):
        """
        num_vars — количество переменных в исходной задаче
        A — матрица коэффициентов ограничений
        signs — список знаков ('<=', '>=', '=')
        b — правая часть ограничений
        c — коэффициенты целевой функции
        sense — 'max' или 'min' (минимизация реализуется через инверсию знаков)
        """
        self.n = num_vars
        self.A = [row[:] for row in A]
        self.signs = signs[:]
        self.b = b[:]
        # Сохраняем оригинальные коэффициенты и направление задачи
        self.orig_c = c[:]
        self.orig_sense = sense
        # Для внутреннего решателя приводим к задаче на максимум (если min — инвертируем)
        self.c = c[:] if sense == 'max' else [-ci for ci in c]
        self.sense = 'max'
        self.m = len(A)
        self.tableau = None
        self.var_names = []
        self.basic_vars = []
        self.artificial_indices = set()

    # =======================================================
    # Приведение задачи к каноническому виду и построение таблицы
    # =======================================================
    def _build_tableau(self):
        m = self.m
        n = self.n
        A = copy.deepcopy(self.A)
        b = self.b[:]

        # Имена переменных для вывода (x1, x2, ...)
        var_names = [f"x{i+1}" for i in range(n)]

        extra_cols_info = []  # хранит информацию о добавленных переменных
        slack_cols = 0
        art_cols = 0

        # Для каждого ограничения определяем, какие переменные добавить
        for i, sign in enumerate(self.signs):
            if sign == '<=':
                var_names.append(f"s{slack_cols+1}")   # добавляем переменную запаса
                extra_cols_info.append(('slack', i))
                slack_cols += 1
            elif sign == '>=':
                var_names.append(f"sr{slack_cols+1}")  # избыточная переменная
                extra_cols_info.append(('surplus', i))
                slack_cols += 1
                var_names.append(f"a{art_cols+1}")     # искусственная переменная
                extra_cols_info.append(('art', i))
                art_cols += 1
            elif sign == '=':
                var_names.append(f"a{art_cols+1}")     # только искусственная переменная
                extra_cols_info.append(('art', i))
                art_cols += 1
            else:
                raise ValueError("Неизвестный знак: " + sign)

        total_vars = n + len(extra_cols_info)
        tableau = [[0.0] * (total_vars + 1) for _ in range(m)]

        # Заполняем матрицу коэффициентов
        for i in range(m):
            for j in range(n):
                tableau[i][j] = A[i][j]
            tableau[i][-1] = b[i]

        # Добавляем slack/surplus/artificial переменные в таблицу
        extra_index = n
        artificial_indices = set()
        for info in extra_cols_info:
            typ, row = info
            if typ == 'slack':
                tableau[row][extra_index] = 1.0
                extra_index += 1
            elif typ == 'surplus':
                tableau[row][extra_index] = -1.0
                extra_index += 1
            elif typ == 'art':
                tableau[row][extra_index] = 1.0
                artificial_indices.add(extra_index)
                extra_index += 1

        # Определяем базисные переменные
        basic_vars = [-1] * m
        for j in range(n, total_vars):
            col = [tableau[i][j] for i in range(m)]
            nonzeros = [i for i, v in enumerate(col) if abs(v) > 1e-9]
            if len(nonzeros) == 1 and abs(col[nonzeros[0]] - 1.0) < 1e-9:
                r = nonzeros[0]
                if basic_vars[r] == -1:
                    basic_vars[r] = j

        # Построение целевой функции для фазы I:
        # минимизируем сумму искусственных переменных → obj = -∑a_j
        obj = [0.0] * (total_vars + 1)
        for j in artificial_indices:
            obj[j] = -1.0

        # Корректировка obj, если искусственные переменные уже в базисе
        for i in range(m):
            bj = basic_vars[i]
            if bj in artificial_indices:
                for k in range(total_vars + 1):
                    obj[k] += tableau[i][k]

        # Сохраняем таблицу и параметры
        self.tableau = tableau
        self.var_names = var_names
        self.basic_vars = basic_vars
        self.artificial_indices = artificial_indices
        self.total_vars = total_vars
        self.phase1_obj = obj

    # =======================================================
    # Элементарная операция — поворот таблицы
    # =======================================================
    def _pivot(self, tab, obj, basic_vars, enter_col, leave_row):
        m = len(tab)
        total_vars = len(tab[0]) - 1
        pivot = tab[leave_row][enter_col]
        if abs(pivot) < 1e-12:
            raise ValueError("Pivot равен нулю!")

        # Нормируем ведущую строку
        for j in range(total_vars + 1):
            tab[leave_row][j] /= pivot

        # Обнуляем столбец
        for i in range(m):
            if i == leave_row:
                continue
            factor = tab[i][enter_col]
            if abs(factor) > 1e-12:
                for j in range(total_vars + 1):
                    tab[i][j] -= factor * tab[leave_row][j]

        # Обновляем целевую функцию
        factor = obj[enter_col]
        if abs(factor) > 1e-12:
            for j in range(total_vars + 1):
                obj[j] -= factor * tab[leave_row][j]

        # Меняем базис
        basic_vars[leave_row] = enter_col

    # =======================================================
    # Выбор входящей переменной (по наибольшему положительному коэф.)
    # =======================================================
    def _choose_entering(self, obj) -> Optional[int]:
        best_val = 0.0
        best_j = None
        for j in range(len(obj) - 1):
            if obj[j] > best_val + 1e-12:
                best_val = obj[j]
                best_j = j
        return best_j

    # =======================================================
    # Выбор выходящей переменной (по минимальному отношению)
    # =======================================================
    def _choose_leaving(self, tab, enter_col) -> Optional[int]:
        m = len(tab)
        min_ratio = None
        leave_row = None
        for i in range(m):
            aij = tab[i][enter_col]
            if aij > 1e-12:
                ratio = tab[i][-1] / aij
                if ratio < -1e-12:
                    continue
                if (min_ratio is None) or (ratio < min_ratio - 1e-12):
                    min_ratio = ratio
                    leave_row = i
        return leave_row

    # =======================================================
    # Общий симплекс-цикл
    # =======================================================
    def _simplex(self, tab, obj, basic_vars, max_iters=5000):
        m = len(tab)
        total_vars = len(tab[0]) - 1
        it = 0
        while True:
            it += 1
            if it > max_iters:
                return ("iter_limit", [], 0.0)

            enter = self._choose_entering(obj)
            if enter is None:
                # Оптимум найден
                val = obj[-1]
                x = [0.0] * total_vars
                for i in range(m):
                    bj = basic_vars[i]
                    if bj is not None and bj < total_vars:
                        x[bj] = tab[i][-1]
                return ("optimal", x, val)

            leave = self._choose_leaving(tab, enter)
            if leave is None:
                # Неограниченная функция
                return ("unbounded", [], 0.0)

            self._pivot(tab, obj, basic_vars, enter, leave)

    # =======================================================
    # Основная функция решения
    # =======================================================
    def solve(self):
        # Фаза I — проверка выполнимости
        self._build_tableau()
        tab = copy.deepcopy(self.tableau)
        obj = self.phase1_obj[:]
        basic_vars = self.basic_vars[:]

        status, x_all, val = self._simplex(tab, obj, basic_vars)

        if status == "unbounded":
            return {"status": "unbounded", "reason": "Phase I unbounded (unexpected)"}

        if status == "iter_limit":
            return {"status": "unknown", "reason": "Phase I iteration limit"}

        # Если оптимум первой фазы != 0, система несовместна
        phase1_val = obj[-1]
        if abs(phase1_val) > 1e-6:
            return {"status": "infeasible", "phase1_obj": phase1_val}

        # Фаза II — решение исходной задачи без искусственных переменных
        total_vars = self.total_vars
        art_idx = sorted(list(self.artificial_indices))
        keep_cols = [j for j in range(total_vars) if j not in art_idx]
        new_total = len(keep_cols)

        new_tab = [[0.0] * (new_total + 1) for _ in range(self.m)]
        for i in range(self.m):
            for new_j, old_j in enumerate(keep_cols):
                new_tab[i][new_j] = tab[i][old_j]
            new_tab[i][-1] = tab[i][-1]

        # Перестраиваем базис
        new_basic = [-1] * self.m
        old_to_new = {old: new for new, old in enumerate(keep_cols)}
        for i in range(self.m):
            old_b = basic_vars[i]
            if old_b in art_idx:
                new_basic[i] = None
            else:
                new_basic[i] = old_to_new.get(old_b, None)

        # Целевая функция для фазы II
        phase2_obj = [0.0] * (new_total + 1)
        for old_j, coef in enumerate(self.c):
            if old_j in old_to_new:
                phase2_obj[old_to_new[old_j]] = coef

        # Корректировка obj с учётом базиса
        for i in range(self.m):
            bj = new_basic[i]
            if bj is not None and bj < new_total:
                cb = phase2_obj[bj]
                if abs(cb) > 1e-12:
                    for j in range(new_total + 1):
                        phase2_obj[j] -= cb * new_tab[i][j]

        # Запускаем симплекс для основной задачи
        status2, x_final, val2 = self._simplex(new_tab, phase2_obj, new_basic)

        if status2 == "optimal":
            # Извлекаем решение только по исходным переменным x1...xn
            solution = [0.0] * self.n
            for old_j in range(self.n):
                if old_j in old_to_new:
                    new_j = old_to_new[old_j]
                    if new_j < len(x_final):
                        solution[old_j] = x_final[new_j]

            # При выводе используем оригинальные коэффициенты, чтобы
            # корректно показать значение целевой функции в исходной постановке
            obj_val = sum(solution[i] * self.orig_c[i] for i in range(self.n))
            return {"status": "optimal", "x": solution, "obj": obj_val}
        else:
            return {"status": status2, "reason": "Phase II " + status2}


# =======================================================
# Разбор входного файла
# =======================================================
def parse_lp_file(text: str):
    lines = [ln.strip() for ln in text.strip().splitlines()
             if ln.strip() and not ln.strip().startswith('#')]

    if not lines:
        raise ValueError("Empty input")

    # Первая строка: max/min и коэффициенты целевой функции
    header = lines[0].split()
    sense = header[0].lower()
    if sense not in ('max', 'min'):
        raise ValueError("First token must be 'max' or 'min'")
    c = [float(x) for x in header[1:]]
    n = len(c)

    A, signs, b = [], [], []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < n + 2:
            raise ValueError("Constraint line has too few tokens: " + ln)
        coeffs = [float(x) for x in parts[:n]]
        rel = parts[n]
        rhs = float(parts[n + 1])
        if rel not in ('<=', '>=', '='):
            raise ValueError("Relation must be one of <=, >=, =. Found: " + rel)
        A.append(coeffs)
        signs.append(rel)
        b.append(rhs)

    return n, A, signs, b, c, sense


# =======================================================
# Точка входа
# =======================================================
def main():
    # Если файл не указан — используется встроенный пример
    if len(sys.argv) < 2:
        print("Usage: python two_phase_simplex.py problem.txt")
        print("No input file provided — solving built-in demo problem.")
        lp_text = """
        max 2 3 1 4
        1 1 1 1 <= 10
        2 1 -1 1 = 8
        0 1 2 1 >= 5
        """
    else:
        fname = sys.argv[1]
        with open(fname, 'r', encoding='utf-8') as f:
            lp_text = f.read()

    # Парсинг входных данных
    try:
        n, A, signs, b, c, sense = parse_lp_file(lp_text)
    except Exception as e:
        print("Ошибка чтения LP файла:", e)
        return

    # Создание и запуск решателя
    solver = TwoPhaseSimplex(n, A, signs, b, c, sense)
    result = solver.solve()

    # Формируем человекочитаемый вывод
    out_lines = []
    out_lines.append("Input LP:")
    out_lines.append(f"sense: {sense}, n = {n}")
    out_lines.append("Constraints:")
    for row, s, rhs in zip(A, signs, b):
        out_lines.append(f"{row} {s} {rhs}")
    out_lines.append(f"c = {c}\n")
    out_lines.append("Result:")

    if result.get("status") == "optimal":
        x = result["x"]
        obj = result["obj"]
        out_lines.append("status: optimal")
        out_lines.append("x = [" + ", ".join(f"{xi:.6g}" for xi in x) + "]")
        out_lines.append(f"Z = {obj:.6g}")
    else:
        out_lines.append("status: " + str(result.get("status")))
        if "reason" in result:
            out_lines.append("reason: " + str(result["reason"]))
        if "phase1_obj" in result:
            out_lines.append("phase1_obj: " + str(result["phase1_obj"]))

    # Вывод в консоль и файл
    out_text = "\n".join(out_lines)
    print(out_text)
    with open("result.txt", "w", encoding="utf-8") as fo:
        fo.write(out_text)
    print("\nРезультат записан в result.txt")


if __name__ == "__main__":
    main()
