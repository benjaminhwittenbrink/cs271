import os 
import numpy as np 

def convert_string_to_ints(row):
    row = row.replace("[", "")
    row = row.replace("]", "")
    vals = row.split(".")
    vals = vals[:-1]
    vals = [int(val.strip()) for val in vals]
    return vals

def calc_weighted_accs(fold, results_file, verbose = False):
    with open(fold + results_file) as f:
        lines = f.readlines()
    
    weighted_accs = []
    for i, l in enumerate(lines):
        if "*** Confusion matrix:" in l: 
            # calculate weighted acc 
            cm_rows = [lines[i + j + 1] for j in range(N_CLASSES)]
            cm_rows_ints = [convert_string_to_ints(row) for row in cm_rows]
            cm = np.array(cm_rows_ints)

            ns = np.sum(cm, 1)
            correct_preds = np.diag(cm)
            acc = correct_preds / ns

            ws = 1 - (ns / ns.sum())
            ws = ws / ws.sum()

            weighted_acc = acc.dot(ws)
            weighted_accs.append(weighted_acc)

            if verbose: print(f"*** Accuracy on the Validation set: {weighted_acc}")
            if verbose: print(l.rstrip())

        else: 
            if verbose: print(l.rstrip())
    
    return weighted_accs
    
    
if __name__ == "__main__":

    N_CLASSES = 3

    res_folders = os.listdir("results/results_infant/")
    res_folders = sorted(res_folders)

    for fold in res_folders:
        folder_path = "results/results_infant/" + fold + "/"
        results = os.listdir(folder_path)


        for results_file in results: 

            if ".txt" in results_file:

                wacc = calc_weighted_accs(folder_path, results_file)
                print(folder_path)
                #print(results_file)
                print(f"Epoch with best weighted acc: {np.argmax(wacc)} with value of {np.max(wacc)}.\n")