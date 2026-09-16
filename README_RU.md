# FHERMA: baseline негaциклического умножения

Задача: `polynomial-multiplication/negacyclic@1.0.0`,
`c = a*b mod (X^N+1, q)`. Основная точка: `N=32768`, `W=868`, `L=28`.

## Реализация

`solve.cu` — точный radix-2 NTT с арифметикой NVIDIA cuPQC BigInt:

1. В `init` находятся примитивный корень `psi` порядка `2N`, обратный корень
   и `N^-1`; на GPU создаются таблицы степеней и постоянные буферы.
2. В `run` входы копируются на GPU, умножаются на `psi^i` и переставляются
   в бит-реверсном порядке.
3. Два прямых NTT выполняются совместно, по одному CUDA-ядру на стадию.
4. Поэлементное произведение переставляется перед обратным NTT.
5. Результат умножается на `psi^-i/N` и копируется на CPU.

Все преобразования и копирования входов/выхода входят в `run`.
Setup не получает операнды. Вычисления точные, без floating point.
`wide_host.h` содержит арифметику поиска корней без внешних зависимостей.
Код поддерживает степени двойки `2 <= N <= 32768` при `W=868`;
на платформе заявлена только соревновательная точка.

Это первоначальный baseline: 33 запуска CUDA-ядер на умножение при `N=32768`,
без объединения стадий и специализированной модульной редукции.

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
Официального GPU-результата пока нет.

## Сабмит

Приватный черновик FHERMA:
https://www.fherma.io/kernels/polynomial-multiplication/ilya-usmanov/ntt-cupqc-baseline

- implementation ID: `6aaabc211be2e96f7342bb92`
- spec ID: `6a87eee7dea7a7548ad4ece4`
- image ID: `6a9833940171486f4faae443` (`cupqc`)
- runner: `6aa6690fd4413d278e91947e` (`fherma-gpu-rtx6000`)
- harness: `cpp`
- challenge: `polynomial-multiplication-2025`

Токены не входят в репозиторий. FHERMA CLI читает локальный
`~/.fherma/config.toml`; GitHub CLI использует свою авторизацию.

## Источники

- https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic
- https://www.fherma.io/docs/cli-implementations
- https://www.fherma.io/docs/submit-a-challenge-solution
- https://github.com/fherma-ai/polynomial-mult-cupqc-example
- https://docs.nvidia.com/cuda/cupqc/libraries/cupqc_bigint/cupqc_bigint_usage.html

Версии scaffold: `fherma==0.2.9`, `fherma-lang==0.14.5`.
