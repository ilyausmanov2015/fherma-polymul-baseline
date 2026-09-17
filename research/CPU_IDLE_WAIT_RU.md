# Ограниченное аппаратное ожидание новых CPU jobs

Эксперимент FHERMA_UMWAIT заменяет длительный idle spin рабочих потоков.
Первые 64 итерации используют PAUSE. Затем на x86 с CPUID WAITPKG worker
вооружает UMONITOR на atomic generation, повторно проверяет acquire-load
и вызывает UMWAIT(C0.1) с пределом 5000 TSC ticks. Публикация generation
с release пробуждает ожидание; повторная проверка после UMONITOR защищает
от потерянного уведомления. Любой spurious wakeup допустим: внешний цикл
снова проверяет generation. Таймаут ограничивает ожидание при shutdown,
когда меняется stop flag, и на системах, игнорирующих UMONITOR.

Быстрое ожидание уже выполняющейся задачи и GPU flags не меняется.
Нет системных настроек частоты, MSR или управления чужими CPU. При
отсутствии CPUID feature остаётся PAUSE. Selective-output-wakeup режим
оставлен на PAUSE, поскольку использует два разных publication address.
Цель — проверить, мешает ли idle spin последовательной работе caller
через SMT или потребление ресурсов; до профиля это гипотеза.

Профиль init дополнительно печатает core_id/package/thread_siblings_list
только для разрешённых CPU. Это диагностика доступной топологии, без
изменения affinity политики. GMP/CPU эмуляция на ARM проверяет fallback;
реальное исполнение WAITPKG должно подтверждаться runner и CPU_IDLE log.

Источники:
https://cdrdv2-public.intel.com/671110/325383-sdm-vol-2abcd.pdf
https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/technical-documentation/monitor-umonitor-performance-guidance.html
https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
