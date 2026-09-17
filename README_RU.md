# FHERMA: baseline негациклического умножения

Задача: `polynomial-multiplication/negacyclic@1.0.0`,
`c = a*b mod (X^N+1, q)`. Основная точка: `N=32768`, `W=868`, `L=28`.

**Официальный результат 17 сентября 2026: 20/20, медиана 0,718 мс —
ускорение 3,53× относительно исходного baseline.**
На проверке 17 сентября, 15:16 МСК: **3-е место из 6**.
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 ГБ.
Исходники: https://github.com/ilyausmanov2015/fherma-polymul-baseline

| Конкурсный запуск | Коммит | Проверено | Медиана | Диапазон |
|---|---|---:|---:|---:|
| Исходный baseline | d736e66 | 20/20 | 2,533 мс | 2,492–2,631 мс |
| Специальная редукция, SoA, fused NTT, pinned, Graph | d8a8d1d | 20/20 | 1,320 мс | 1,274–1,416 мс |
| Два atomic-потока копирования, прямой выход | 48d4d1b | 20/20 | 0,952 мс | 0,909–1,992 мс |
| Восемь потоков, prefault выхода, все стадии fused, компактные корни | f1171aa | 20/20 | 0,718 мс | 0,688–1,145 мс |

Два дополнительных прогревочных случая в каждом запуске также прошли,
но не входят в score. CUDA 12.8.61, cuPQC 0.6.0.

[Новый полный запуск](https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic/runs/6aabd90ce3bdd1a5530b1a48),
[исходный запуск](https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic/runs/6aaabf591be2e96f7342c7d1),
[лидерборд](https://www.fherma.io/kernels/polynomial-multiplication/challenges/polynomial-multiplication-2025?tab=leaderboard).
Измерения промежуточных вариантов: `results/experiments.json` и `EXPERIMENTS_RU.md`.
Текущий HEAD может содержать экспериментальные изменения; последний полностью
подтверждённый конкурсный коммит — `f1171aabd6ef548d570f2aaa8201aac8722ff942`.

## Реализация

`solve.cu` — точный radix-2 NTT с арифметикой NVIDIA cuPQC BigInt:

1. В `init` находятся примитивный корень `psi` порядка `2N`, обратный корень
   и `N^-1`; на GPU создаются таблицы степеней и постоянные буферы.
2. В `run` входы копируются на GPU, умножаются на `psi^i` и переставляются
   в бит-реверсном порядке.
3. Два прямых NTT выполняются совместно; первые восемь стадий объединены в shared memory.
4. Поэлементное произведение переставляется перед обратным NTT.
5. Результат умножается на `psi^-i/N` и копируется на CPU.

Все преобразования и копирования входов/выхода входят в `run`.
Setup не получает операнды. Вычисления точные, без floating point.
`wide_host.h` содержит арифметику поиска корней без внешних зависимостей.
Код поддерживает степени двойки `2 <= N <= 32768` при `W=868`;
на платформе заявлена только соревновательная точка.

При `N=32768` вычисление содержит 9 CUDA-ядер вместо исходных 33.
Широкое произведение cuPQC редуцируется двумя сворачиваниями для `q=2^868-c`.
Внутренний формат SoA обеспечивает совместное чтение соседними потоками.
Первые восемь и последние семь стадий NTT объединены отдельно;
между ними выполняется coalesced-транспонирование. Для обоих наборов стадий
корни заранее упакованы в соответствующую раскладку.
Обмен использует постоянные pinned-буферы; передачи и ядра захвачены CUDA Graph.
CPU-копирование входов восемью потоками входит в тайминг. Выход передаётся DMA
сразу в собственный std::vector; во время GPU-вычисления готовится следующий
вектор. Его выделение, параллельная подготовка страниц, заполнение и регистрация
также входят в тайминг.
Перед возвратом результата регистрация снимается. В последнем полном прогоне среднее 0,780 мс при медиане 0,718 мс;
максимум снизился с примерно 2 до 1,145 мс. Следующие эксперименты продолжают
оптимизировать обмен памятью и GPU-арифметику.

## Сборка на GPU

Нужен официальный образ `fherma/cupqc` с cuPQC BigInt и CUDA >=12.8.
`CUPQC_SDK_DIR` должен указывать на SDK. Архитектура выбирается по GPU
на машине сборки (`-arch=native`), включая RTX PRO 6000 Blackwell.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/solution /path/to/point-directory
```

`main.cpp` и `fherma.h` получены официальным CLI и не изменены.
`CMakeLists.txt` адаптирован для архитектуры runner; `fherma.toml` не заменяет
выбор `cpp` и `cupqc` при регистрации реализации на платформе.

## Проверки на Mac

```sh
/Users/alfa/codes/venv/bin/python3.14 tests/build_emulation.py
/Users/alfa/codes/venv/bin/python3.14 tests/validate.py --n 32 --binary build-emulation/solution
/Users/alfa/codes/venv/bin/python3.14 tests/validate.py --n 32768 --binary build-emulation/solution
```

Для этих проверок нужны GMP C++ headers/library и Python `gmpy2==2.3.1`.
Тестовый адаптер компилирует тела CUDA-ядер из `solve.cu` с GMP вместо cuPQC
и последовательно исполняет все блоки/потоки. Соревновательная сборка
не использует этот адаптер.

Независимый эталон — умножение больших целых с упаковкой коэффициентов
(Kronecker substitution), затем свёртка `X^N=-1` и редукция по `q`.
Основание упаковки больше `N*(q-1)^2`, поэтому переносы не смешивают
коэффициенты. На малом размере эталон дополнительно сравнивается с
квадратичным умножением.

Проверены 18 случаев: 9 при `N=32` и 9 при `N=32768`, всегда `W=868`.
В каждой группе: два плотных случайных входа, нулевой множитель, единица,
переход через `X^N`, максимальные коэффициенты, чередующиеся коэффициенты,
границы 32-битных слов и повторный случайный вход на том же состоянии.
Все результаты совпали побайтно. Отчёты: `results/local-n32.json` и
`results/local-n32768.json`.

**Это проверка алгебры и индексации, не проверка CUDA/cuPQC на GPU.**
CPU-время эмуляции не является результатом для лидерборда.
Официальные GPU-измерения приведены в начале документа.

## Сабмит

Приватная карточка реализации FHERMA (результат участвует в лидерборде):
https://www.fherma.io/kernels/polynomial-multiplication/ilya-usmanov/ntt-cupqc-baseline

- implementation ID: `6aaabc211be2e96f7342bb92`
- spec ID: `6a87eee7dea7a7548ad4ece4`
- image ID: `6a9833940171486f4faae443` (`cupqc`)
- runner: `6aa6690fd4413d278e91947e` (`fherma-gpu-rtx6000`)
- harness: `cpp`
- challenge: `polynomial-multiplication-2025`

Токены не входят в репозиторий. FHERMA CLI читает локальный
`~/.fherma/config.toml`; GitHub CLI использует свою авторизацию.

После публикации точного коммита в доступном runner репозитории:

```sh
python tools/fherma_submit.py status
python tools/fherma_submit.py history
python tools/fherma_submit.py attach --repository https://github.com/ilyausmanov2015/fherma-polymul-baseline
python tools/fherma_submit.py benchmark --seeds 2
# После успешной проверки сборки и результата:
python tools/fherma_submit.py enter
python tools/fherma_submit.py run RUN_ID
```

`attach` по умолчанию закрепляет текущий `HEAD`. Для приватного GitHub-репозитория
нужен отдельный read-only clone token в форме FHERMA. Общий токен GitHub CLI
автоматически в FHERMA не передаётся.
Команды `status`, `attach`, `benchmark`, `enter` и `history` проверены на платформе.
Карточка реализации переключается на точный коммит каждого эксперимента.
Текущий указатель можно проверить через `status`; конкурсный результат связан
с конкретной ревизией и не заменяется обычным трёхслучайным бенчмарком.

## Источники

- https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic
- https://www.fherma.io/docs/cli-implementations
- https://www.fherma.io/docs/submit-a-challenge-solution
- https://github.com/fherma-ai/polynomial-mult-cupqc-example
- https://docs.nvidia.com/cuda/cupqc/libraries/cupqc_bigint/cupqc_bigint_usage.html

Версии scaffold: `fherma==0.2.9`, `fherma-lang==0.14.5`.
