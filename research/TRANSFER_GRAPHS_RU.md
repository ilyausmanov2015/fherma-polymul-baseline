# Вызовы CUDA и графы передачи

Исходный лучший 9ed2a18: CPU упаковывает четыре отдельные пары A/B.
Для каждой части в run вызываются две cudaMemcpyAsync, EventRecord,
StreamWaitEvent и GraphLaunch подготовки. После главного GraphLaunch
добавляются четыре D2H и четыре EventRecord.

0782ec5 захватывает две H2D и внешний EventRecord в transfer_graph;
prepare_graph начинается с внешнего EventWait. В run остаются два
GraphLaunch на часть. Обе очереди сохраняются, буферы частей независимы.
Результат: 3/3 +2/2, 480,093 мкс — хуже подтверждённого baseline.

488f3d3 дополнительно включает четыре D2H и внешние сигналы output_ready
в главный граф. CPU ждёт каждый сигнал перед чтением соответствующей части.
Ни захват, ни upload графа не исполняют вычисление входов в init.
Результат 488f3d3: 3/3 +2/2, 470,672 мкс; CUDA CI успешно.

Механизм внешних событий между графами документирован NVIDIA:
https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html
https://docs.nvidia.com/cuda/archive/12.8.0/pdf/CUDA_Runtime_API.pdf

d82b5a9 возвращает обычные события и объединяет A/B одной части в
cudaMemcpy2DAsync: height=2, width=count*4, spitch=count*4, dpitch=words*4.
CPU pipeline гарантирует смежность A/B внутри каждой части. Данные
копируются в две непересекающиеся области GPU ABI-буфера.
https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__MEMORY.html

Дополнительные измерения: native RNS с 24 входными и 8 выходными потоками
2ed9af6 — 475,504 мкс; две части ввода и вывода c4aad59 — 519,640 мкс.
Все 3/3, дополнительного выигрыша нет.

Измерение d82b5a9: 3/3 +2/2, медиана 460,893 мкс; разница с
коротким baseline 461,668 мкс слишком мала для вывода об ускорении.

UVA-пробы на четырёхкомпонентном backend: c7c4160 читает paired pinned
input непосредственно в transpose kernel — 681,875 мкс, 3/3.
2a37001 пишет CRT непосредственно в pinned host_output и затем обычный
владеющий vector — 524,035 мкс, 3/3. Обе версии медленнее DMA.
Перед GPU обе прошли полный N32768 integer oracle.

Параметры runner: pageableMemoryAccess=0,
pageableMemoryAccessUsesHostPageTables=0, concurrentManagedAccess=1,
directManagedMemAccessFromHost=0. Поэтому HMM для обычных malloc-входов
здесь недоступен; mapped UVA использован только для cudaMallocHost-буферов.
Это разные возможности CUDA:
https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__UNIFIED.html
https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html
