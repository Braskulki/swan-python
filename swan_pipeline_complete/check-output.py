from pathlib import Path

import numpy as np
from scipy.io import loadmat


OUTPUT_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "processed"
    / "output.mat"
)


def describe_variable(name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)

    finite_mask = np.isfinite(values)
    finite_count = int(np.count_nonzero(finite_mask))
    total_count = int(values.size)
    nan_count = int(np.count_nonzero(np.isnan(values)))

    print(f"\nVAR: {name}")
    print(f"shape: {values.shape}")
    print(f"pontos totais: {total_count}")
    print(f"pontos válidos: {finite_count}")
    print(f"NaN: {nan_count}")
    print(f"percentual válido: {100 * finite_count / total_count:.2f}%")

    if finite_count == 0:
        print("Nenhum valor finito encontrado.")
        return

    valid = values[finite_mask]

    print(f"min: {np.min(valid):.4f}")
    print(f"max: {np.max(valid):.4f}")
    print(f"média: {np.mean(valid):.4f}")
    print(f"mediana: {np.median(valid):.4f}")
    print(f"zeros válidos: {np.count_nonzero(valid == 0)}")


def main() -> None:
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {OUTPUT_FILE}"
        )

    data = loadmat(OUTPUT_FILE)

    variables = {
        name: values
        for name, values in data.items()
        if not name.startswith("__")
    }

    print(f"Arquivo: {OUTPUT_FILE}")
    print(f"Quantidade de variáveis: {len(variables)}")

    for name, values in variables.items():
        describe_variable(name, values)


if __name__ == "__main__":
    main()