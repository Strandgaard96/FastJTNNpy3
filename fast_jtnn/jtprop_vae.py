import copy

import rdkit.Chem as Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import DataStructs
from rdkit.Chem import AllChem

from .chemutils import (
    attach_mols,
    copy_edit_mol,
    enum_assemble,
    is_valid_smiles,
    set_atommap,
)
from .datautils_prop import tensorize_prop
from .jtmpn import JTMPN
from .jtnn_dec import JTNNDecoder
from .jtnn_enc import JTNNEncoder
from .mol_tree import MolTree
from .mpn import MPN
from .nnutils import create_var


class JTpropVAE(nn.Module):
    def __init__(
        self,
        vocab,
        hidden_size,
        latent_size,
        depthT,
        depthG,
        denticity="monodentate",
        dropout=0,
    ):
        super(JTpropVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size = (
            latent_size // 2
        )  # Tree and Mol has two vectors

        self.jtnn = JTNNEncoder(
            hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size)
        )
        self.decoder = JTNNDecoder(
            vocab, hidden_size, latent_size, nn.Embedding(vocab.size(), hidden_size)
        )

        self.jtmpn = JTMPN(hidden_size, depthG)
        self.mpn = MPN(hidden_size, depthG)

        self.A_assm = nn.Linear(latent_size, hidden_size, bias=False)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        self.T_mean = nn.Linear(hidden_size, latent_size)
        self.T_var = nn.Linear(hidden_size, latent_size)
        self.G_mean = nn.Linear(hidden_size, latent_size)
        self.G_var = nn.Linear(hidden_size, latent_size)

        self.propNN = nn.Sequential(
            nn.Linear(self.latent_size * 2, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 2),
            nn.Dropout(dropout),
        )
        self.prop_loss = nn.MSELoss()
        self.denticity = denticity
        self.dropout = dropout

    def encode(self, jtenc_holder, mpn_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        return tree_vecs, tree_mess, mol_vecs

    def encode_latent_from_smiles(self, smiles_list, props):
        tree_batch = []
        for i, elem in enumerate(smiles_list):
            s = MolTree(elem)
            tree_batch.append(s)
        _, _, jtenc_holder, mpn_holder = tensorize_prop(
            tree_batch, props, self.vocab, assm=False
        )
        z1, z2 = self.encode_latent(jtenc_holder, mpn_holder)
        return z1, z2

    def encode_latent(self, jtenc_holder, mpn_holder):
        tree_vecs, _ = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        tree_var = -torch.abs(self.T_var(tree_vecs))
        mol_var = -torch.abs(self.G_var(mol_vecs))
        return torch.cat([tree_mean, mol_mean], dim=1), torch.cat(
            [tree_var, mol_var], dim=1
        )

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))  # Following Mueller et al.
        kl_loss = (
            -0.5
            * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var))
            / batch_size
        )
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def sample_prior(self, prob_decode=False):
        z_tree = torch.randn(1, self.latent_size).cuda()
        z_mol = torch.randn(1, self.latent_size).cuda()
        return self.decode(z_tree, z_mol, prob_decode)

    def optimize(
        self,
        x_batch,
        sim_cutoff,
        lr=2.0,
        num_iter=20,
        type="both",
        prob_decode=False,
        minimize=True,
    ):
        x_batch, x_prop, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)

        mol = Chem.MolFromSmiles(x_batch[0].smiles)
        fp1 = AllChem.GetMorganFingerprint(mol, 2)

        z_tree_mean = self.T_mean(x_tree_vecs)
        -torch.abs(self.T_var(x_tree_vecs))  # Following Mueller et al.
        z_mol_mean = self.G_mean(x_mol_vecs)
        -torch.abs(self.G_var(x_mol_vecs))  # Following Mueller et al.

        mean = torch.cat([z_tree_mean, z_mol_mean], dim=1)
        cur_vec = create_var(mean.data, True)

        visited = []
        cuda0 = torch.device("cuda:0")
        first_predicted = []
        second_predicted = []
        lr_homo = 1.5 * lr
        for step in range(num_iter):
            prop_val = self.propNN(cur_vec)
            # print(prop_val)
            first_predicted.append(prop_val.tolist()[0][0])
            second_predicted.append(prop_val.tolist()[0][1])
            # grad = torch.autograd.grad(prop_val, cur_vec,grad_outputs=torch.ones_like(prop_val))[0]
            dydx3 = torch.tensor([], dtype=torch.float32, device=cuda0)
            for i in range(2):
                li = torch.zeros_like(prop_val)
                li[:, i] = 1.0
                d = torch.autograd.grad(
                    prop_val, cur_vec, retain_graph=True, grad_outputs=li
                )[
                    0
                ]  # dydx: (batch_size, input_dim)
                dydx3 = torch.concat((dydx3, d.unsqueeze(dim=1)), dim=1)

            dydx3 = dydx3.squeeze()

            scaler = -1 if minimize else 1

            # Normalize gradient
            norm0 = torch.nn.functional.normalize(dydx3[0], dim=-1)
            norm1 = torch.nn.functional.normalize(dydx3[1], dim=-1)
            # nnorm = torch.linalg.vector_norm(norm)

            # Get the gradient magnitudes
            # n0 = torch.linalg.vector_norm(dydx3[0])
            # n1 = torch.linalg.vector_norm(dydx3[1])

            if type == "both":
                cur_vec = cur_vec.data + scaler * lr * norm1 + scaler * lr_homo * norm0
            elif type == "first":
                cur_vec = cur_vec.data + scaler * lr_homo * norm0
            elif type == "second":
                cur_vec = cur_vec.data + scaler * lr * norm1
            elif type == "first_second":
                cur_vec = cur_vec.data + scaler * lr_homo * norm0 - scaler * lr * norm1
            # elif type == "second_first":
            #     cur_vec = cur_vec.data + scaler * lr * norm1 - scaler * lr_homo * norm0
            else:
                raise ValueError

            cur_vec = create_var(cur_vec, True)
            visited.append(cur_vec)

        # Now we want to get the best possible vectors.

        tanimoto = []
        li, r = 0, num_iter - 1
        counter = 1
        gradient_idx = []
        smiles_list = []
        while li < r - 1:
            mid = (li + r) // 2
            new_vec = visited[mid]
            tree_vec, mol_vec = torch.chunk(new_vec, 2, dim=1)
            new_smiles = self.decode(tree_vec, mol_vec, prob_decode=prob_decode)
            if new_smiles is None:
                r = mid - 1
                continue

            new_mol = Chem.MolFromSmiles(new_smiles)
            fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            # print(f'Tanimoto score {mid}:', sim)
            tanimoto.append((counter, sim))
            gradient_idx.append((counter, mid))
            smiles_list.append((counter, new_smiles))
            if sim < sim_cutoff:
                r = mid - 1
            else:
                li = mid
            counter += 1

        tree_vec, mol_vec = torch.chunk(visited[li], 2, dim=1)
        # tree_vec,mol_vec = torch.chunk(best_vec, 2, dim=1)
        new_smiles = self.decode(tree_vec, mol_vec, prob_decode=prob_decode)

        tanimoto_candidates = [
            (x[1], s[1], grad_idx[1])
            for x, s, grad_idx in zip(tanimoto, smiles_list, gradient_idx)
            if x[1] > sim_cutoff
        ]
        tanimoto_candidates.sort(reverse=True, key=lambda x: x[0])
        tanimoto_candidates = set(tanimoto_candidates)
        if not is_valid_smiles(new_smiles):
            for tan, sm, grad_idx in tanimoto_candidates:
                if is_valid_smiles(sm):
                    new_smiles = sm
                    break
                else:
                    print("noo  not valid smiles")
                    new_smiles = None
        # Print all the smiles along the gradient.
        # to_print = []
        # for v in visited[0:selected_idx]:
        #     tree_vec, mol_vec = torch.chunk(v, 2, dim=1)
        #     new_smiles = self.decode(tree_vec, mol_vec, prob_decode=prob_decode)
        #
        #     prop_val = self.propNN(v)
        #     # print(prop_val)
        #     prediction = prop_val.tolist()
        #
        #     new_mol = Chem.MolFromSmiles(new_smiles)
        #     # Chem.AssignStereochemistry(new_mol)
        #     fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
        #     sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        #     # print(sim,x_batch[0].smiles,new_smiles)
        #
        #     to_print.append((new_smiles, prediction, sim))
        # print(to_print)

        if new_smiles is None:
            return x_batch[0].smiles, 1.0
        new_mol = Chem.MolFromSmiles(new_smiles)
        fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        if sim >= sim_cutoff:
            return new_smiles, sim
        else:
            print("Noooo cutoooof")
            return x_batch[0].smiles, 1.0

    def forward(self, x_batch, beta):
        x_batch, x_prop, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        # Extract prop for later

        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
        z_tree_vecs, tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs, mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)

        kl_div = tree_kl + mol_kl
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_batch, z_tree_vecs)
        assm_loss, assm_acc = self.assm(
            x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess
        )

        if self.dropout == 1:
            # Learn properties
            all_vec = torch.cat([z_tree_vecs, z_mol_vecs], dim=1)
            prop_label = create_var(x_prop)
            prop_loss = self.prop_loss(self.propNN(all_vec).squeeze(), prop_label)

        return (
            word_loss + topo_loss + assm_loss + beta * kl_div + prop_loss,
            kl_div.item(),
            word_acc,
            topo_acc,
            assm_acc,
            prop_loss.item(),
        )

    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, x_tree_mess):
        jtmpn_holder, batch_idx = jtmpn_holder
        fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
        batch_idx = create_var(batch_idx)

        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)

        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)
        x_mol_vecs = self.A_assm(x_mol_vecs)  # bilinear
        scores = torch.bmm(x_mol_vecs.unsqueeze(1), cand_vecs.unsqueeze(-1)).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(mol_batch):
            comp_nodes = [
                node
                for node in mol_tree.nodes
                if len(node.cands) > 1 and not node.is_leaf
            ]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]))
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label))

        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode):
        # currently do not support batch decoding
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        pred_root, pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode)
        if len(pred_nodes) == 0:
            return None
        elif len(pred_nodes) == 1:
            return pred_root.smiles

        # Mark nid & is_leaf & atommap
        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = len(node.neighbors) == 1
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder, mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _, tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (
            tree_mess,
            mess_dict,
        )  # Important: tree_mess is a matrix, mess_dict is a python dict

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze()  # bilinear

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol, _ = self.dfs_assemble(
            tree_mess,
            x_mol_vecs,
            pred_nodes,
            cur_mol,
            global_amap,
            [],
            pred_root,
            None,
            prob_decode,
            check_aroma=True,
        )
        if cur_mol is None:
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {
                atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()
            }
            cur_mol, pre_mol = self.dfs_assemble(
                tree_mess,
                x_mol_vecs,
                pred_nodes,
                cur_mol,
                global_amap,
                [],
                pred_root,
                None,
                prob_decode,
                check_aroma=False,
            )
            if cur_mol is None:
                cur_mol = pre_mol

        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None

    def dfs_assemble(
        self,
        y_tree_mess,
        x_mol_vecs,
        all_nodes,
        cur_mol,
        global_amap,
        fa_amap,
        cur_node,
        fa_node,
        prob_decode,
        check_aroma,
    ):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands, aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles, cand_amap = list(zip(*cands))
        aroma_score = torch.Tensor(aroma_score).cuda()
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
            cand_vecs = self.jtmpn(
                fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0]
            )
            scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = torch.Tensor([1.0])

        if prob_decode:
            try:
                probs = (
                    F.softmax(scores.view(1, -1), dim=1).squeeze() + 1e-7
                )  # prevent prob = 0
                cand_idx = torch.multinomial(probs, probs.numel())
            except Exception:
                _, cand_idx = torch.sort(scores, descending=True)
        else:
            _, cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][
                    ctr_atom
                ]

            cur_mol = attach_mols(
                cur_mol, children, [], new_global_amap
            )  # father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue

            has_error = False
            for nei_node in children:
                if nei_node.is_leaf:
                    continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(
                    y_tree_mess,
                    x_mol_vecs,
                    all_nodes,
                    cur_mol,
                    new_global_amap,
                    pred_amap,
                    nei_node,
                    cur_node,
                    prob_decode,
                    check_aroma,
                )
                if tmp_mol is None:
                    has_error = True
                    if i == 0:
                        pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error:
                return cur_mol, cur_mol

        return None, pre_mol
