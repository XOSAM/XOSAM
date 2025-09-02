import math

def min_folds(target_height, t0=0.00008):
    n = 0
    thickness = t0
    while thickness < target_height:
        n += 1
        thickness = t0 * (2 ** n)
    return n

mt_fuji_height = 3776
folds_fuji = min_folds(mt_fuji_height)
print(f"Mt. Fuji requires {folds_fuji} folds.")

def min_folds_log(target_height, t0=0.00008):
    return math.ceil(math.log2(target_height / t0))

mt_fuji = 3776
moon = 3.84e8
proxima = 4.0175e16

print("\nMinimum folds required:")
print("Mt. Fuji:", min_folds_log(mt_fuji), "folds")
print("Moon:", min_folds_log(moon), "folds")
print("Proxima Centauri:", min_folds_log(proxima), "folds")

def paper_length(n, t0=0.00008):
    return (math.pi * t0 / 6) * ((2**n + 4) * (2**n - 1))

print("\nPaper length requirements:")
for name, target in [("Mt. Fuji", mt_fuji), ("Moon", moon), ("Proxima Centauri", proxima)]:
    n = min_folds_log(target)
    L = paper_length(n)
    print(f"{name}: {n} folds, requires paper length {L:.2e} meters")
