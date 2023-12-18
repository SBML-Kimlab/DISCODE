import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import esm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

def listup_salient_residues(attention_matrix,):
    att_sum = np.sum(np.sum(np.sum(attention_matrix, axis=0), axis=0), axis=0)
    average, std = np.mean(att_sum), np.std(att_sum)
    threshold = average + 2 * std
    idx = np.where((att_sum > threshold) == True)[0]
    return idx

def collect_attention_weights(inputs, model):
    x = inputs
    attention_weights = []
    for i in range(len(model.transformer_encoder.layers)):
        _, weight = model.transformer_encoder.layers[i].self_attn(x, x, x, average_attn_weights=False)
        attention_weights.append(weight.squeeze(0).cpu().numpy())
        x = model.transformer_encoder.layers[i](x)
    attention_weights = np.asarray(attention_weights)
    return attention_weights

def make_mut_candidate(idx, name, sequence,):
    name_dict = {}
    name_split = name.split("_")
    if len(name_split) == 1:
        mut_list = []
        for index in idx:
            aa_list = list("ACDEFGHIKLMNPQRSTVWY")
            original = sequence[index]
            if original in aa_list:
                aa_list.remove(original)
            for i in range(len(aa_list)):
                mut_list.append(name + "_" + original + str(index + 1) + aa_list[i])
    else:
        mut_list = []
        for index in idx:
            aa_list = list("ACDEFGHIKLMNPQRSTVWY")
            for aa in aa_list:
                name_dict = {}
                for i in range(1, len(name_split)):
                    name_dict[int(name_split[i][1:-1]) - 1] = name_split[i]
                if index in name_dict.keys():
                    continue
                name_dict[index] = sequence[index] + str(index + 1) + aa 
                name_items = sorted(name_dict.items())
                x = name_split[0]
                for item in name_items:
                    x += "_" + item[-1]
                mut_list.append(x)
    return mut_list

def tokenize_and_dataloader(name, sequence):
    data = [(name, sequence)]
    esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    if torch.cuda.is_available() == True:
        esm_model = esm_model.cuda()
    batch_labels, _, batch_tokens = batch_converter(data)
    if torch.cuda.is_available() == True:
        batch_tokens = batch_tokens.cuda()
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[len(esm_model.layers)], return_contacts=False)
        token_representations = results["representations"][len(esm_model.layers)].squeeze(0) 
    dataloader = DataLoader([[token_representations[1:-1], batch_labels[0]]])
    return dataloader

def replace_sequence(mut, sequence):
    mut_list = mut.split("_")[1:]
    seq_list = list(sequence)
    for mut_candidate in mut_list:
        ori_aa, pos, mut_aa = mut_candidate[0], int(mut_candidate[1:-1]), mut_candidate[-1]
        assert seq_list[pos-1] == ori_aa
        seq_list[pos-1] = mut_aa
    seq = "".join(seq_list)
    return seq

def model_prediction(dataloader, model):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            wt_prob = model(inputs).cpu().squeeze(0)
            wt_label = (wt_prob >= 0.5).float()
            attention_weights = collect_attention_weights(inputs, model)
        original_idx = listup_salient_residues(attention_weights,)
    return original_idx, wt_prob, wt_label, labels, attention_weights

def make_df_sorting_by_prob(candidate, wt_label):
    if (wt_label.numpy() == np.array([1, 0])).all() == True:
        label = "NAD"
    elif (wt_label.numpy() == np.array([0, 1])).all() == True:
        label = "NADP"
    index, prob = [], []
    for i in range(len(candidate)):
        index.append(candidate[i][0][0])
        prob.append(candidate[i][1].numpy())
    df = pd.DataFrame(prob, columns=["NAD", "NADP"], index=index)
    if label == "NAD":
        df = df.sort_values(by=["NADP"], ascending=False)
    elif label == "NADP":
        df = df.sort_values(by=["NAD"], ascending=False)
    return df

def make_max_attention_map(attention_weights):
    max_attn = np.max(np.max(attention_weights, axis=-1), axis=-1)
    plt.figure(figsize=(10,4))
    sns.heatmap(max_attn, cmap="Blues")

def plot_attention_sum(attention_weights, idx, sequence):
    att_sum = np.sum(np.sum(np.sum(attention_weights, axis=0), axis=0), axis=0)
    average, std = np.mean(att_sum), np.std(att_sum)
    threshold = average + 2 * std
    plt.plot(np.arange(1, len(att_sum) + 1), att_sum)
    plt.plot((1, len(att_sum) + 1), (threshold, threshold), color="red", linestyle="--")
    print(f"The maximum attention sum is ... {np.max(att_sum):.3f}")
    salient_residues = []
    for res in idx:
        salient_residues.append(sequence[res] + str(res+1))
    print(f"The salient residues are ... {salient_residues}")
    for i in range(len(salient_residues)):
        print(f"The attention sum of {salient_residues[i]} is ... {att_sum[idx[i]]:.3f}")

def scan_switch_mutation(model, sequence, name="unknwon", pickle_path=".", max_num_mutation=3, max_num_solution=20, prob_thres=0.5, mode="iterative_num"):
    wt_dataloader = tokenize_and_dataloader(name, sequence)
    wt_idx, wt_prob, wt_label, wt_name, _ = model_prediction(wt_dataloader, model)
    print(f"The wildtype label probability is ...{wt_prob}")
    convert_dict = {}
    index_dict = {}
    if mode == "iterative_prob":
        for i in range(max_num_mutation):
            results = {"Convert" : {}, "No" : {}}
            results["No"][i] = {}
            if i == 0:
                generate_mutation(model, wt_label, wt_idx, i, wt_name[0], sequence, results, index_dict, mode)
            else:
                mut_keys = list(index_dict.keys())
                for key in mut_keys:
                    if len(key.split("_")) == i + 1:
                        generate_mutation(model, wt_label, index_dict[key], i, key, sequence, results, index_dict, mode)
            print(f"The mutation step {wt_name, i + 1} end...")
            with open(pickle_path + "/" + name + "_iterative_prob_mutation_" + str(i + 1) + ".pkl", "wb") as f:
                pkl.dump(results, f)
                f.close()
            for key in results["Convert"].keys():
                convert_dict[key] = results["Convert"][key]
        if len(convert_dict.keys()) == 0:
            print(f"The mutation was not found...")
        else:
            keys = convert_dict.keys()
            values = convert_dict.values()
            df = pd.DataFrame(values, index=keys, columns=["NAD", "NADP"])
            if (wt_label == torch.tensor([1,0])).sum().item() == 2:
                df = df.sort_values(by="NADP", ascending=False)
            elif (wt_label == torch.tensor([0,1])).sum().item() == 2:
                df = df.sort_values(by="NAD", ascending=False)
            if len(df) > max_num_solution:
                df = df.loc[df.index[:max_num_solution]]
            return df
    elif mode == "iterative_num":
        for i in range(max_num_mutation):
            results = {"Convert" : {}, "No" : {}}
            results["No"][i] = {}
            if i == 0:
                generate_mutation(model, wt_label, wt_idx, i, wt_name[0], sequence, results, index_dict, mode)
            else:
                mut_keys = list(index_dict.keys())
                for key in mut_keys:
                    if len(key.split("_")) == i + 1:
                        generate_mutation(model, wt_label, index_dict[key], i, key, sequence, results, index_dict, mode)
            print(f"The mutation step {wt_name, i + 1} end...")
            with open(pickle_path + "/" + name + "_iterative_num_mutation_" + str(i + 1) + ".pkl", "wb") as f:
                pkl.dump(results, f)
                f.close()
            for key in results["Convert"].keys():
                convert_dict[key] = results["Convert"][key]
            if len(convert_dict) > 0:
                print(f"The mutation was found in {i + 1}step")
                break
        if len(results["Convert"].keys()) == 0:
            print(f"The mutation was not found...")
        else:
            keys = convert_dict.keys()
            values = convert_dict.values()
            df = pd.DataFrame(values, index=keys, columns=["NAD", "NADP"])
            if (wt_label == torch.tensor([1,0])).sum().item() == 2:
                df = df.sort_values(by="NADP", ascending=False)
            elif (wt_label == torch.tensor([0,1])).sum().item() == 2:
                df = df.sort_values(by="NAD", ascending=False)
            if len(df) > max_num_solution:
                df = df.loc[df.index[:max_num_solution]]      
            return df
    elif mode == "shortest":
        results = {"Convert" : {}, "No" : {}}
        for i in range(max_num_mutation):
            results["No"][i] = {}
            if i == 0:
                generate_mutation(model, wt_label, wt_idx, i, wt_name[0], sequence, results, index_dict, mode)
                if len(results["Convert"].keys()) > 0:
                    print(f"The mutation was derived in {i + 1} mutations. Iteration stopped.")
                    break
            else:
                sorted_results = sorted(results["No"][i-1].items(), key = lambda item: item[1], reverse=True)
                generate_mutation(model, wt_label, index_dict[sorted_results[0][0]], i, sorted_results[0][0], sequence, results, index_dict, mode)
                if len(results["Convert"].keys()) > 0:
                    print(f"The mutation was derived in {i + 1} mutations. Iteration stopped.")
                    break
        with open(pickle_path + "/" + name + "_shortest_mutation_" + str(i + 1) + ".pkl", "wb") as f:
            pkl.dump(results, f)
            f.close()
        if len(results["Convert"].keys()) == 0:
            print(f"The mutation was not found... Please use iterative mode.")
        else:
            keys = results["Convert"].keys()
            values = results["Convert"].values()
            df = pd.DataFrame(values, index=keys, columns=["NAD", "NADP"])
            if (wt_label == torch.tensor([1,0])).sum().item() == 2:
                df = df.sort_values(by="NADP", ascending=False)
            elif (wt_label == torch.tensor([0,1])).sum().item() == 2:
                df = df.sort_values(by="NAD", ascending=False)
            return df
    else:
        print("The mode command is unknown.. Please check the mode argument and rerun.")
        
def generate_mutation(model, wt_label, idx, trial, name, sequence, results, index_dict, mode):
    if mode == "shortest":
        mut_list = make_mut_candidate(idx, name, sequence)
        for mut in mut_list:
            mut_seq = replace_sequence(mut, sequence)
            mut_dataloader = tokenize_and_dataloader(mut, mut_seq)
            mut_idx, mut_prob, mut_label, mut_index, _ = model_prediction(mut_dataloader, model)
            if (wt_label == mut_label).sum().item() == 0:
                index_dict[mut_index[0]] = mut_idx
                results["Convert"][mut_index[0]] = mut_prob.numpy()
            elif (wt_label == torch.tensor([1,0])).sum().item() == 2:
                index_dict[mut_index[0]] = mut_idx
                results["No"][trial][mut_index[0]] = float(mut_prob[1])
            elif (wt_label == torch.tensor([0,1])).sum().item() == 2:
                index_dict[mut_index[0]] = mut_idx
                results["No"][trial][mut_index[0]] = float(mut_prob[0])
    else:
        mut_list = make_mut_candidate(idx, name, sequence)
        for mut in mut_list:
            if mut not in index_dict.keys():
                mut_seq = replace_sequence(mut, sequence)
                mut_dataloader = tokenize_and_dataloader(mut, mut_seq)
                mut_idx, mut_prob, mut_label, mut_index, _ = model_prediction(mut_dataloader, model)
                if (wt_label == mut_label).sum().item() == 0:
                    index_dict[mut_index[0]] = mut_idx
                    results["Convert"][mut_index[0]] = mut_prob.numpy()
                else:
                    index_dict[mut_index[0]] = mut_idx
                    results["No"][trial][mut_index[0]] = mut_prob