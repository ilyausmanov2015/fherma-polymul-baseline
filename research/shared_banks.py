"""Model 32 four-byte shared-memory banks; this is not a timing model."""
from collections import Counter

for radix, stages in [(4, [1, 4, 16, 64, 256]), (8, [1, 8, 64])]:
    def swizzle(index):
        if radix == 8:
            high = (index >> 5) & 7
            return index ^ high ^ ((high & 6) << 2)
        return index ^ (((index >> 5) & 3) * 5) ^ ((index >> 2) & 16)

    assert sorted(swizzle(i) for i in range(1024)) == list(range(1024))
    for half in stages:
        before, after = [], []
        for warp in range(1024 // radix // 32):
            for register in range(radix):
                addresses = []
                for thread in range(warp * 32, (warp + 1) * 32):
                    j = thread & (half - 1)
                    addresses.append(radix * (thread - j) + j + register * half)
                before.append(max(Counter(i % 32 for i in addresses).values()))
                after.append(max(Counter(swizzle(i) % 32 for i in addresses).values()))
        assert max(after) == 1
        print(f"radix={radix} half={half}: multiplicity {max(before)} -> {max(after)}")
