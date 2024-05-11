import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import rdkit
import rdkit.Chem as Chem
import torch
import torch.nn as nn
from rdkit.Chem import Descriptors
from torch.autograd import Variable

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))
from fast_jtnn import JTpropVAE, MolTreeFolder_prop, Vocab
from fast_molopt.preprocess_prop import main_preprocess

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def parse_args(arg_list: list = None) -> argparse.Namespace:
    """Parse arguments from command line.

    Args:
        arg_list (list, optional): Automatically obtained from the command line if provided.
        If no arguments are given but default arguments are defined, the latter are used.

    Returns:
        argparse.Namespace: Dictionary-like class that contains the driver arguments.
    """
    parser = argparse.ArgumentParser(
        description="Strip substitute atoms from generated ligands"
    )

    parser.add_argument("--training_path", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", default="optimize-processed")
    parser.add_argument("--prop_path", default=True)
    parser.add_argument("--nsplits", type=int, default=2)

    parser.add_argument("--hidden_size", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_size", type=int, default=56)
    parser.add_argument("--depthT", type=int, default=20)
    parser.add_argument("--depthG", type=int, default=3)
    parser.add_argument("--cutoff", type=float, default=0.2)

    parser.add_argument(
        "--type", type=str, default="first", help="Which property to optimize on"
    )

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--clip_norm", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--step_beta", type=float, default=0.002)
    parser.add_argument("--max_beta", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=500)

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--anneal_rate", type=float, default=0.9)
    parser.add_argument("--anneal_iter", type=int, default=1000)
    parser.add_argument("--kl_anneal_iter", type=int, default=3000)
    parser.add_argument("--print_iter", type=int, default=50)
    parser.add_argument("--save_iter", type=int, default=1000)
    parser.add_argument(
        "--denticity",
        choices=["monodentate", "bidentate"],
        type=str,
        default="monodentate",
    )

    return parser.parse_args(arg_list)


def main():
    opts = parse_args()

    vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
    vocab = Vocab(vocab)

    hidden_size = int(opts.hidden_size)
    latent_size = int(opts.latent_size)
    sim_cutoff = float(opts.cutoff)

    model = JTpropVAE(
        vocab,
        int(hidden_size),
        int(latent_size),
        int(opts.depthT),
        int(opts.depthG),
        denticity=opts.denticity,
    ).cuda()
    print(model)
    model.load_state_dict(torch.load(opts.model_path))
    model = model.cuda()

    # filePath = "/home/magstr/git/xyz2mol_tm/jt_vae_research/dft_labeled_data/region_production/region_samples_production.txt"
    # if os.path.exists(filePath):
    #     os.remove(filePath)
    # filePath = "/home/magstr/git/xyz2mol_tm/jt_vae_research/dft_labeled_data/region_production/region_samples_prop_production.txt"
    # if os.path.exists(filePath):
    #     os.remove(filePath)

    # Process dataframe
    # df_og = pd.read_csv(
    #     "/home/magstr/git/xyz2mol_tm/jt_vae_research/dft_labeled_data/region_production/10_samples_each_region_labeled_set.csv"
    # )
    # for i, row in df_og.iterrows():
    #     with open(
    #         "/home/magstr/git/xyz2mol_tm/jt_vae_research/dft_labeled_data/region_production/region_samples_production.txt",
    #         "a",
    #     ) as f:
    #         f.write(f"{row['sub_smi']}\n")
    #     with open(
    #         "/home/magstr/git/xyz2mol_tm/jt_vae_research/dft_labeled_data/region_production/region_samples_prop_production.txt",
    #         "a",
    #     ) as f:
    #         f.write(f"{row['homo-lumo']},{row['Ir-cm5']}\n")

    df_ligands = pd.read_csv(opts.training_path, header=None, names=["Enriched SMILES"])
    df_props = pd.read_csv(
        opts.prop_path, header=None, names=["HOMO-LUMO gap (Eh)", "Ir charge"]
    )
    input_df = pd.concat([df_ligands, df_props], axis=1)

    output_dir = Path(f"opt_{time.strftime('%Y%m%d-%H%M%S')}")
    output_dir.mkdir(exist_ok=True)

    # Preprocess data
    # train_path = "/home/magstr/git/xyz2mol_tm/jt_vae_research/dft_labeled_data/region_production/region_samples_production.txt"
    # prop_path = "/home/magstr/git/xyz2mol_tm/jt_vae_research/dft_labeled_data/region_production/region_samples_prop_production.txt"
    main_preprocess(opts.training_path, opts.prop_path, opts.output_path, opts.nsplits)

    loader = MolTreeFolder_prop(
        opts.output_path,
        vocab,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        optimize=True,
    )

    defaultdict(list)

    with open(output_dir / "opts.json", "w") as file:
        json.dump(vars(opts), file)

    directions = [
        ("first", "maximize"),
        ("first", "minimize"),
        ("both", "maximize"),
        ("both", "minimize"),
        ("second", "maximize"),
        ("second", "minimize"),
        ("first_second", "maximize"),
        ("first_second", "minimize"),
    ]

    for i, batch in enumerate(loader):
        for dir in directions:
            current_type = dir[0]
            smiles = batch[0][0].smiles
            print(i, smiles)

            batch[1].numpy().squeeze()

            Chem.MolFromSmiles(smiles)
            minimize = True if dir[1] == "minimize" else False
            new_smiles, sim = model.optimize(
                batch,
                sim_cutoff=sim_cutoff,
                lr=opts.lr,
                num_iter=100,
                type=current_type,
                prob_decode=False,
                minimize=minimize,
            )

            Chem.MolFromSmiles(new_smiles)
            # results["new_smiles"].append(new_smiles)
            # results["sim"].append(sim)
            # results["type"].append(current_type)
            # results["minimize"].append(minimize)

            # Write a row to a csv file.
            with open(output_dir / "optimize_results.csv", "a") as f1:
                writer = csv.writer(
                    f1,
                    delimiter=",",
                    lineterminator="\n",
                )
                # Get the row elements of the original data
                r = input_df.iloc[i].to_list()
                # Append the data from optimized row
                list_of_props = r + [smiles, new_smiles, sim, current_type, minimize]
                writer.writerow(list_of_props)


if __name__ == "__main__":
    main()
