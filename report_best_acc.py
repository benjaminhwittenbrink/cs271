import os 
import numpy as np 

def get_accs(fold, results_file, verbose = False):
    with open(fold + results_file) as f:
        lines = f.readlines()
    
    accs = []
    for i, l in enumerate(lines):
        if "*** Accuracy on the Validation set:" in l: 
            # calculate weighted acc 
            acc_string = l.split(":")[-1]
            acc = float(acc_string)
            accs.append(acc)

            if verbose: print(l.rstrip())
        else: 
            if verbose: print(l.rstrip())
    
    return accs
    
    
if __name__ == "__main__":

    N_CLASSES = 3
    FOLDER = "results/results_bmi_quint/"

    res_folders = os.listdir(FOLDER)
    res_folders = sorted(res_folders)

    for fold in res_folders:
        folder_path = FOLDER + fold + "/"
        results = os.listdir(folder_path)


        for results_file in results: 

            if ".txt" in results_file:

                acc = get_accs(folder_path, results_file)
                print(folder_path)
                #print(results_file)
                print(f"Epoch with best weighted acc: {np.argmax(acc)} with value of {np.max(acc)}.\n")