import sys

sys.path.append("../")
import os
import sys
from multiprocessing import Pool
from optparse import OptionParser
from pathlib import Path

import numpy as np
import rdkit
from tqdm import tqdm

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))
from fast_jtnn import *
from fast_jtnn.datautils_prop import *


def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree


def convert(train_path, prop_path, pool, num_splits, output_path):
    out_path = os.path.join(output_path, "./")
    if os.path.isdir(out_path) is False:
        os.makedirs(out_path)

    with open(train_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]
    print("Input File read")

    with open(prop_path) as f:
        prop_data = [line.strip("\r\n ").split(",") for line in f]
    # Convert to float
    # ## Start debug
    # temp_prop = []
    # for x in prop_data:
    #     sub =[]
    #     try:
    #         assert len(x)==2
    #         sub = [float(y) for y in x]
    #         temp_prop.append(sub)
    #     except Exception as e:
    #         print(x)
    #         print(e)
    # prop_data = temp_prop
    # ## End debug
    # prop_data = [[float(y) for y in x] for x in prop_data]
    print("Prop file read")

    # Verify that the number of properties match the number of data points
    if len(data) != len(prop_data):
        raise Exception(
            "TEMRINATED, number of properties does not match number of data points"
        )

    print("Tensorizing .....")
    all_data = pool.map(tensorize, data)
    all_data_split = np.array_split(all_data, num_splits)
    prop_data_split = np.array_split(prop_data, num_splits)
    print("Tensorizing Complete")

    print("Adding prop to data")

    for split_id in tqdm(list(range(num_splits)), position=0, leave=True):
        with open(os.path.join(output_path, "tensors-%d.pkl" % split_id), "wb") as f:
            pickle.dump((all_data_split[split_id], prop_data_split[split_id]), f)

    return True


def main_preprocess(
    train_path, prop_path, output_path, num_splits=10, njobs=os.cpu_count()
):
    pool = Pool(njobs)
    convert(train_path, prop_path, pool, num_splits, output_path)
    return True


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-p", "--prop_path", dest="prop_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    parser.add_option("-o", "--output", dest="output_path")

    opts, args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)
    convert(opts.train_path, opts.prop_path, pool, num_splits, opts.output_path)
