import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt 


def convert_string_to_ints(row):
    row = row.replace("[", "")
    row = row.replace("]", "")
    vals = row.split(".")
    vals = vals[:-1]
    vals = [int(val.strip()) for val in vals]
    return vals


if __name__ == "__main__":

    cm_strings = {
        # """
        #     [[333. 163.  57.]
        #     [141. 194.  76.]
        #     [ 11.  57.  52.]]
        # """,
        # ViT mean bmi bin 
        "vit_Mean_BMI_bin": 
            """
            [[ 57.  41.   2.]
            [110. 468. 148.]
            [ 12.  83. 163.]]
            """,
        # ViT mean bmi bin w / country embeddings 
        "vit_country_Mean_BMI_bin": 
            """
            [[ 88.  20.   4.]
            [147. 336. 250.]
            [ 12.  46. 181.]]
            """,
        "vit_Mean_BMI_bin_quint":
        """
        [[178.  34.  22.  10.  15.]
        [ 75.  47.  39.  11.  29.]
        [ 37.  40.  51.  23.  57.]
        [ 25.  21.  43.  41.  88.]
        [  7.  10.  24.  21. 136.]]
        """, 
        "vit_under5_mortality_bin_quint":
        """
        [[111.  29.  33.  18.  26.]
        [ 88.  38.  28.  21.  40.]
        [ 59.  35.  44.  16.  59.]
        [ 34.  31.  37.  23. 106.]
        [ 19.  10.  23.  21. 135.]]
        """,
        "vit_country_under5_mortality_bin":
        """
        [[333. 163.  57.]
        [141. 194.  76.]
        [ 11.  57.  52.]]
        """,
    }
    titles = {
        "vit_Mean_BMI_bin": "BMI CDC binning (ViT)",
        "vit_country_Mean_BMI_bin": "BMI CDC binning (ViT w/ country embedding)",
        "vit_Mean_BMI_bin_quint": "BMI quintile binning (ViT)",
    }
    plt.rcParams.update({'font.size': 22})

    for cm_key in cm_strings: 
        print(cm_key)
        cm = np.array(convert_string_to_ints(cm_strings[cm_key]))
        dim = int(np.sqrt(len(cm)))
        cm = np.reshape(cm, (dim,dim))

        fig, ax = plt.subplots(1,1, figsize=(7,6))
        sns.heatmap(cm, annot=True, fmt='g', cmap = "Blues", annot_kws={"size": 20})
        ax.set_ylabel("True class")
        ax.set_xlabel("Predicted class")
        #plt.title(titles[cm_key])
        plt.savefig("output/heatmaps/" + cm_key + ".png")
        plt.show()

        denoms = [np.sum(cm[0:i]) + np.sum(cm[i+1:]) for i in range(cm.shape[0])]
        nums = [np.sum(cm[0:i, i]) + np.sum(cm[i+1:, i]) for i in range(cm.shape[0])]
        fpr = np.array(nums) / np.array(denoms) 

        print("TNR by class:", np.round(1 - fpr, 2))

        cm = cm / cm.sum(1)[:, None]
        fig, ax = plt.subplots(1,1, figsize=(7,6))
        sns.heatmap(np.round(cm,2), annot=True, fmt='g', cmap = "Blues", annot_kws={"size": 20})
        ax.set_ylabel("True class")
        ax.set_xlabel("Predicted class")
        #plt.title(titles[cm_key])
        plt.savefig("output/heatmaps/" + cm_key + "_rownorm.png")
        plt.show()

        print("TPR by class:", np.diag(np.round(cm, 2)))
