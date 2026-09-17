# WC output и non-temporal CPU loads

FHERMA_OUTPUT_WC выделяет выходной staging с cudaHostAllocWriteCombined.
D2H и completion flags сохраняются. После подтверждения каждой части
CPU workers читают её MOVNTDQA (AVX-512, затем SSE4.1 fallback) и пишут
обычные cached stores в свежий владеющий std::vector. Простой memcpy
остаётся переносимым fallback, но на настоящей WC-памяти может быть медленным.

Гипотеза: GPU не придётся инвалидировать много CPU cache lines staging,
оставшихся после предыдущего результата. Это отдельная проверка поверх
fe5fc67 с прежними 16/8 потоками и без изменения арифметики; улучшение
не предполагается до измерения. Размещение physical-first выключено.

MFENCE на каждом worker после acquire готовности и перед WC reads
упорядочивает чтение относительно GPU-публикации. Второй MFENCE перед
CPU acknowledgment завершает чтение и запись результата. Это существенно:
WC и streaming load buffers не дают обычных гарантий WB-кэша. Инициализация
пустых WC staging в init также завершается MFENCE. Input/output data
обрабатываются только в run. Соседние части не перезаписываются до
полного завершения предыдущего run; никакие результаты не переиспользуются.

Векторные загрузки выровнены на 64 (AVX-512) или 16 (SSE4.1) байт.
Невыровненные head/tail копируются отдельно, без чтения за пределами.
Тест 1300 комбинаций размеров/offsets проверяет все copy-пути и guards;
Linux CI исполняет доступный ISA путь. Обычная память теста не эмулирует
WC-свойства; публикацию и повторные вызовы проверяет реальный GPU benchmark.

Источники: Intel SDM, MOVNTDQA (включая требование MFENCE при других
агентах записи), и CUDA Programming Guide, Write-Combining Memory:
https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-2b-manual.pdf
https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html
