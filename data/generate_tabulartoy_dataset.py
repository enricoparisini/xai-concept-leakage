import os
import sys
import argparse

from xai_concept_leakage.data.tabulartoy_auxiliary import generate_tabulartoy_data


def main():
    parser = argparse.ArgumentParser(description="Generate and save TabularToy data.")
    parser.add_argument("delta", type=float, help="Delta parameter (e.g. 0.25).")
    parser.add_argument(
        "n_samples", type=int, help="Total number of samples to generate (e.g. 10000)."
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default=None,
        help=(
            "Optional output folder. If omitted, it will be constructed as "
            "data/TabularToy/tabulartoy_{int(delta*100)}_{int(n_samples//1000)}k/"
        ),
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    master_folder = os.path.abspath(os.path.join(script_dir, os.pardir))
    sys.path.insert(0, master_folder)

    data_folder = os.path.join(master_folder, "data")

    # Build default save_folder if not provided
    if args.save_folder is None:
        save_folder = os.path.join(
            data_folder,
            "TabularToy",
            f"tabulartoy_{int(args.delta * 100)}_{int(args.n_samples // 1000)}k/",
        )
    else:
        save_folder = args.save_folder

    # Ensure directory exists
    os.makedirs(save_folder, exist_ok=True)

    (x_train, x_val, x_test, c_train, c_val, c_test, y_train, y_val, y_test) = (
        generate_tabulartoy_data(args.delta, args.n_samples, save_folder)
    )

    print(
        f"Generated TabularToy data with delta={args.delta}, n_samples={args.n_samples}"
    )
    print(f"Saved to: {save_folder}")


if __name__ == "__main__":
    main()
