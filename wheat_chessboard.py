import numpy as np
import matplotlib.pyplot as plt

def wheat_chessboard(n, m):
    indices = np.arange(n * m, dtype=np.uint64)
    board = 2 ** indices
    return board.reshape(n, m)

board_8x8 = wheat_chessboard(8, 8)
print("8x8 Chessboard (grains per square):")
print(board_8x8)

row_sums = board_8x8.sum(axis=1)
col_sums = board_8x8.sum(axis=0)
print("\nRow sums:", row_sums)
print("Column sums:", col_sums)

col_avgs = col_sums / 8
print("\nColumn averages:", col_avgs)

plt.bar(range(1, 9), col_avgs)
plt.xlabel("Column")
plt.ylabel("Average grains")
plt.title("Average Wheat Grains per Column (8x8 Board)")
plt.show()

board_32x32 = wheat_chessboard(32, 32)

plt.imshow(board_32x32, cmap="hot", interpolation="nearest", norm="log")
plt.colorbar(label="Grains (log scale)")
plt.title("Wheat Distribution on 32x32 Chessboard")
plt.show()

def wheat_chessboard_loop(n, m):
    board = np.zeros((n, m), dtype=np.uint64)
    count = 1
    for i in range(n):
        for j in range(m):
            board[i, j] = count
            count *= 2
    return board
