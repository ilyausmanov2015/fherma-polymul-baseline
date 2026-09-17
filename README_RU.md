# FHERMA: baseline негациклического умножения

Задача: `polynomial-multiplication/negacyclic@1.0.0`,
`c = a*b mod (X^N+1, q)`. Основная точка: `N=32768`, `W=868`, `L=28`.

**Официальный результат 17 сентября 2026: 20/20, медиана 0,387 мс —
ускорение 6,55× относительно исходного baseline.**
На проверке 17 сентября, 20:16 МСК: **3-е место из 6**.
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
| H2D и перевод в RNS одновременно, coalesced CRT output | 122a5b2 | 20/20 | 0,474 мс | 0,453–0,496 мс |
| 16 потоков CPU на входе, 8 на выходе | 9ed2a18 | 20/20 | 0,464 мс | 0,456–0,488 мс |
| Четыре компоненты, CRT512, radix-8, перестановка shared memory | a89e75f | 20/20 | 0,454 мс | 0,437–0,475 мс |
| Fused product, warp-tail, отдельные CPU acknowledgments | 9d717d4 | 20/20 | 0,421 мс | 0,402–0,431 мс |
| DIF без bit reversal, основной CPU участвует в выводе | 4a5662f | 20/20 | 0,412 мс | 0,403–0,433 мс |
| Групповая подготовка входа, lazy 30-bit NTT | 1f78057 | 20/20 | 0,406 мс | 0,388–0,427 мс |
| Paired H2D и восемь частей вывода | 2e1e217 | 20/20 | 0,394 мс | 0,379–0,410 мс |
| Fenced stream writes для уведомления CPU о готовности D2H | 09dca35 | 20/20 | 0,387 мс | 0,376–0,411 мс |

Два дополнительных прогревочных случая в каждом запуске также прошли,
но не входят в score. CUDA 12.8.61, cuPQC 0.6.0.

[Новый полный запуск](https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic/runs/6aac1f86b615118421bfa186),
[исходный запуск](https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic/runs/6aaabf591be2e96f7342c7d1),
[лидерборд](https://www.fherma.io/kernels/polynomial-multiplication/challenges/polynomial-multiplication-2025?tab=leaderboard).
Измерения промежуточных вариантов: `results/experiments.json` и `EXPERIMENTS_RU.md`.
Текущий HEAD может содержать экспериментальные изменения; последний полностью
подтверждённый конкурсный коммит — `09dca35dea1a5bc70dcfd7b511c574a931567c11`.

## Реализация

Лучший подтверждённый backend — `quartic/solve.cu`:

1. Каждый 868-битный коэффициент представлен четырьмя компонентами по
   217 бит. Умножение проводится в кольце Y^4=c, затем Y=2^217.
2. В init выбираются 16 простых p<2^30 (при необходимости p<2^31), для которых существуют корни
   порядка 2N и четыре корня Y^4=c. Точные границы гарантируют однозначный
   знаковый CRT всех четырёх компонент. Создаются параметры и пустые буферы.
3. В run 16 CPU-потоков упаковывают вход четырьмя частями; DMA следующей
   части перекрывается с преобразованием предыдущей в 64 канала остатков.
   Четыре warp используют общую плитку входных коэффициентов для четырёх
   простых; отдельное транспонирование входа больше не нужно.
4. Два прямых NTT выполняются совместно в порядке DIF: отдельная
   bit-reversal перестановка не требуется. В блоке 1024 значения три стадии
   объединяются за проход (radix-8). Перестановка shared memory исключает
   конфликты банков в этих проходах. Хвост объединён с транспонированием; butterfly обмениваются регистрами
   через warp shuffle после одной синхронизации shared memory.
5. Поэлементное произведение объединено с загрузкой первого блока обратного NTT.
   Преобразование даёт остатки точной свёртки.
   Обратная 4-точечная DFT и cuPQC BigInt512 восстанавливают четыре знаковые
   компоненты; cuPQC BigInt896 собирает итог по q без floating point.
6. Выход передаётся восемью частями; 8 CPU-потоков читают каждую часть
   только после fenced completion flag и копируют в обычный владеющий std::vector.
   После D2H CUDA пишет номер запуска в mapped pinned flag; CPU ждёт его
   acquire-load. При отсутствии поддержки используются CUDA events.

Все преобразования, копирования, выделение обычного выходного std::vector
и ожидания входят в run. Буферы отдельных частей не перекрываются;
на исключении GPU-stream синхронизируется до возврата управления.
Код поддерживает степени двойки 2<=N<=32768, W=868, q=2^868-c, 0<c<2^28;
на платформе заявлена только соревновательная точка.

Полный прогон 09dca35: медиана 386,7625 мкс, среднее 389,045 мкс,
диапазон 375,579–410,964 мкс. Короткий прогон — 382,421 мкс, 3/3.
Отчёт: `results/best-challenge.json`. Границы CRT и журнал:
`research/QUARTIC_RNS_RU.md`, `research/RNS_PLAN_RU.md`, `EXPERIMENTS_RU.md`.

Предыдущий `rns/solve.cu` использует 57 простых и компактный CRT928.
Его лучший полный результат — 464,3965 мкс (9ed2a18).

Альтернативный `solve.cu` выполняет NTT непосредственно по 868-битному q
через cuPQC BigInt896. Он использует две pseudo-Mersenne редукции, SoA,
объединение первых восьми и последних семи стадий. Его лучший полный
результат — 701,431 мкс, коммит 20f262e. Эксперимент Карацубы для него
документирован в `research/KARATSUBA_RU.md`.

Backend выбирается параметром CMake `-DFHERMA_SOLVER=rns`, `wide` или `quartic`.
Текущий HEAD может содержать очередной эксперимент; для воспроизведения
конкурсного результата используйте точный коммит 09dca35 из начала документа.

## Сборка на GPU

Нужен официальный образ `fherma/cupqc` с cuPQC BigInt и CUDA >=12.8.
`CUPQC_SDK_DIR` должен указывать на SDK. Архитектура выбирается по GPU
на машине сборки (`-arch=native`), включая RTX PRO 6000 Blackwell.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DFHERMA_SOLVER=quartic
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
