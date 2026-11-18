import matplotlib.pyplot as plt

import algorithms as algorithms
from game import create_board

DEPTH_KWARG = "depth"
SIMS_KWARG = "num_simulations"

DEPTHS = [1, 2, 3, 4, 5]
SIMULATIONS = [100, 500, 1000, 2000]
SEED = 42


def benchmark(board, algorithm, **kwargs):
    res = algorithm(board, **kwargs)
    return {"name": algorithm.__name__, "move": res["move"], "evaluation_score": res["score"],
            "nodes_explored": res["nodes_explored"], "time": res["time"]}


def run_all():
    """Lance toutes les configurations et renvoie une liste de lignes."""
    rows = []
    board = create_board()

    for algo in (algorithms.minimax, algorithms.alphabeta):
        for d in DEPTHS:
            r = benchmark(board, algo, **{DEPTH_KWARG: d})
            r["config"] = f"Depth {d}"
            r["param"] = d
            rows.append(r)

    for n in SIMULATIONS:
        r = benchmark(board, algorithms.mcts, **{SIMS_KWARG: n, "seed": SEED})
        r["config"] = f"{n} simulations"
        r["param"] = n
        rows.append(r)

    return rows


def verify(rows):
    """Vérifie : même valeur Minimax, et Alpha-Beta explore <= nœuds que Minimax."""
    mm = {r["param"]: r for r in rows if r["name"] == "minimax"}
    ab = {r["param"]: r for r in rows if r["name"] == "alphabeta"}
    for d in DEPTHS:
        same = mm[d]["evaluation_score"] == ab[d]["evaluation_score"]
        fewer = ab[d]["nodes_explored"] <= mm[d]["nodes_explored"]
        print(f"Profondeur {d} : même valeur = {same} | "
              f"Alpha-Beta <= Minimax en nœuds = {fewer}")


def print_table(rows):
    print(f"\n{'Algorithm':<12}{'Configuration':<20}{'Move':>6}{'Score':>12}{'Nodes':>12}{'Time (s)':>12}")
    print("-" * 74)
    for r in rows:
        print(f"{r['name']:<12}{r['config']:<20}{r['move'] + 1:>6}"
              f"{r['evaluation_score']:>12.3f}{r['nodes_explored']:>12}{r['time']:>12.4f}")


def plot_results(rows):
    mm = [r for r in rows if r["name"] == "minimax"]
    ab = [r for r in rows if r["name"] == "alphabeta"]
    mc = [r for r in rows if r["name"] == "mcts"]

    plt.figure()
    plt.plot([r["param"] for r in mm], [r["nodes_explored"] for r in mm], "o-", label="Minimax")
    plt.plot([r["param"] for r in ab], [r["nodes_explored"] for r in ab], "s-", label="Alpha-Beta")
    plt.yscale("log")
    plt.xlabel("Search depth")
    plt.ylabel("Nodes explored (log scale)")
    plt.title("Nodes explored vs depth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.savefig("nodes_vs_depth.png", dpi=150, bbox_inches="tight")

    plt.figure()
    plt.plot([r["param"] for r in mc], [r["time"] for r in mc], "o-", label="MCTS")
    plt.xlabel("Number of simulations")
    plt.ylabel("Execution time (s)")
    plt.title("MCTS execution time vs simulations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.savefig("mcts_time.png", dpi=150, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    results = run_all()
    verify(results)
    print_table(results)
    plot_results(results)