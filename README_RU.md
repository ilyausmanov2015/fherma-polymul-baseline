# FHERMA: baseline негациклического умножения

Задача: `polynomial-multiplication/negacyclic@1.0.0`,
`c = a*b mod (X^N+1, q)`. Основная точка: `N=32768`, `W=868`, `L=28`.

**Официальный результат 17 сентября 2026: 20/20, медиана 0,514 мс —
ускорение 4,93× относительно исходного baseline.**
На проверке 17 сентября, 16:20 МСК: **3-е место из 6**.
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 ГБ.
Исходники: https://github.com/ilyausmanov2015/fherma-polymul-baseline

| Конкурсный запуск | Коммит | Проверено | Медиана | Диапазон |
|---|---|---:|---:|---:|
| Исходный baseline | d736e66 | 20/20 | 2,533 мс | 2,492–2,631 мс |
| Специальная редукция, SoA, fused NTT, pinned, Graph | d8a8d1d | 20/20 | 1,320 мс | 1,274–1,416 мс |
| Два atomic-потока копирования, прямой выход | 48d4d1b | 20/20 | 0,952 мс | 0,909–1,992 мс |
| Восемь потоков, prefault выхода, все стадии fused, компактные корни | f1171aa | 20/20 | 0,718 мс | 0,688–1,145 мс |
| Ввод и вывод четырьмя частями с перекрытием CPU/GPU | 20f262e | 20/20 | 0,701 мс | 0,655–0,751 мс |
| RNS, radix-4, точный CRT1024, перекрытие ввода/вывода | e064b52 | 20/20 | 0,597 мс | 0,576–0,633 мс |
| CRT928, объединённый tail-transpose, cached CPU output | d155c4f | 20/20 | 0,514 мс | 0,503–0,532 мс |

Два дополнительных прогревочных случая в каждом запуске также прошли,
но не входят в score. CUDA 12.8.61, cuPQC 0.6.0.

[Новый полный запуск](https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic/runs/6aabec9ee3bdd1a5530bfb81),
[исходный запуск](https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic/runs/6aaabf591be2e96f7342c7d1),
[лидерборд](https://www.fherma.io/kernels/polynomial-multiplication/challenges/polynomial-multiplication-2025?tab=leaderboard).
Измерения промежуточных вариантов: `results/experiments.json` и `EXPERIMENTS_RU.md`.
Текущий HEAD может содержать экспериментальные изменения; последний полностью
подтверждённый конкурсный коммит — `d155c4f6142b9cf88a5a94b39d958a9026053cba`.

## Реализация

Лучший подтверждённый backend — `rns/solve.cu`:

1. В init выбираются 57 простых p<2^31 с корнями порядка 2N. Их произведение
   P строго больше 4*N*(q-1)^2; граница проверяется точными целыми числами.
   Создаются только параметры, таблицы и пустые рабочие буферы.
2. В run CPU и GPU передают входы четырьмя частями с перекрытием. Коэффициенты
   на GPU транспонируются по 32-битным словам и переводятся в остатки по p.
3. Два прямых NTT выполняются совместно. Первые десять стадий объединены
   попарно (radix-4) в shared memory; последние пять используют отдельную
   транспонированную раскладку и упакованные корни.
4. Поэлементное произведение и обратный NTT дают остатки точной целочисленной
   негациклической свёртки.
5. cuPQC BigInt928 накапливает взвешенные CRT-остатки. Целая fixed-point
   поправка с доказанной границей восстанавливает результат по q без
   floating point и без вероятностных допущений.
6. Выход передаётся четырьмя частями. CPU читает каждую часть только после
   её CUDA-события; обычный memcpy может идти одновременно со следующей DMA.

Все преобразования, копирования, выделение обычного выходного std::vector
и ожидания входят в run. Буферы отдельных частей не перекрываются;
на исключении GPU-stream синхронизируется до возврата управления.
Код поддерживает степени двойки 2<=N<=32768, W=868, q=2^868-c, 0<c<2^28;
на платформе заявлена только соревновательная точка.

Полный прогон d155c4f: медиана 513,930 мкс, среднее 514,986 мкс,
диапазон 502,782–531,737 мкс. Короткий прогон — 519,225 мкс, 3/3.
Отчёт: `results/best-challenge.json`. Границы CRT и журнал:
`research/RNS_PLAN_RU.md`, `EXPERIMENTS_RU.md`.

Альтернативный `solve.cu` выполняет NTT непосредственно по 868-битному q
через cuPQC BigInt896. Он использует две pseudo-Mersenne редукции, SoA,
объединение первых восьми и последних семи стадий. Его лучший полный
результат — 701,431 мкс, коммит 20f262e. Эксперимент Карацубы для него
документирован в `research/KARATSUBA_RU.md`.

Backend выбирается параметром CMake `-DFHERMA_SOLVER=rns` или `wide`.
Текущий HEAD может содержать очередной эксперимент; для воспроизведения
конкурсного результата используйте точный коммит d155c4f из начала документа.

## Сборка на GPU

Нужен официальный образ `fherma/cupqc` с cuPQC BigInt и CUDA >=12.8.
`CUPQC_SDK_DIR` должен указывать на SDK. Архитектура выбирается по GPU
на машине сборки (`-arch=native`), включая RTX PRO 6000 Blackwell.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DFHERMA_SOLVER=rns
cmake --build build -j
./build/solution /path/to/point-directory
```

`main.cpp` и `fherma.h` получены официальным CLI и не изменены.
`CMakeLists.txt` адаптирован для архитектуры runner; `fherma.toml` не заменяет
выбор `cpp` и `cupqc` при регистрации реализации на платформе.

## Проверки на Mac

```sh
/Users/alfa/codes/venv/bin/python3.14 tests/build_emulation.py --source rns/solve.cu
/Users/alfa/codes/venv/bin/python3.14 tests/validate.py --n 32 --source rns/solve.cu --binary build-emulation-rns/solution
/Users/alfa/codes/venv/bin/python3.14 tests/validate.py --n 32768 --source rns/solve.cu --binary build-emulation-rns/solution
```

Для этих проверок нужны GMP C++ headers/library и Python `gmpy2==2.3.1`.
Тестовый адаптер компилирует тела CUDA-ядер выбранного backend с GMP вместо cuPQC
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
Все результаты совпали побайтно. Дополнительно RNS проверен на N=2 (9/9).
`results/local-n32.json` и `results/local-n32768.json` сохраняют исходный
широкий baseline; актуальные локальные отчёты создаются в `local/n*-w868/`.

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
