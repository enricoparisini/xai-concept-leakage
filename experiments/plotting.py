import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from experiments.experiment_utils import extract_scores_from_results

##################################################################################################
### Plot intervention performance and leakage scores:
##################################################################################################

def plot_intervention_performance(results, policy="random", checkpoint_names=None, 
                                  palette_name=None, y_lim=None, save_path=None,
                                  title=None, linewidth=1, title_fontsize=16, legend_pos="auto", 
                                  add_markers = True):
    scores_dict = extract_scores_from_results(results, score_labels=[policy])
    reps, n_int = scores_dict[next(iter(scores_dict.keys()))].shape
    if checkpoint_names is None:
        checkpoint_names = list(scores_dict.keys())
    palette = sns.color_palette(palette_name, len(checkpoint_names)) 
    df_list = []
    
    for i_int in range(n_int):
        df_temp = pd.DataFrame()
        list_scores = []
        list_model_names = []
        list_names = []
        for model_name in checkpoint_names:
            scores = scores_dict[model_name][:, i_int]
            if reps == 1:
                scores = [scores.item()]
            else:
                scores = list(scores)
            list_scores += scores
            list_model_names += [model_name]*reps
            list_names += [i_int]*reps   
        df_temp["Model"] = list_model_names  
        df_temp["# interventions"] = list_names
        df_temp[r"$y_{acc}$"] = list_scores
        df_list.append(df_temp)
    df = pd.concat(df_list)

    sns.set_context("paper", font_scale=1.7, rc={"font.size":1,"axes.titlesize":1,"axes.labelsize":20, 
                                                 "legend.fontsize":16, "legend.title_fontsize":16})
    ax = sns.lineplot(data=df, x='# interventions', y=r"$y_{acc}$", hue='Model', 
                      linewidth=linewidth, palette=palette, estimator='mean', errorbar='ci')  
    # Add markers at mean values
    if add_markers:
        for idx, model_name in enumerate(checkpoint_names):
            model_df = df[df["Model"] == model_name]
            means = model_df.groupby("# interventions")[r"$y_{acc}$"].mean()
            plt.scatter(means.index, means.values, color=palette[idx], label=None, marker='o', s=50, zorder=5)
    
    if legend_pos != "auto":
        sns.move_legend(ax, "lower left", bbox_to_anchor=(legend_pos[0], legend_pos[1]), title='Model')
    if y_lim is not None:
        ax.set(ylim=(0, y_lim))
    _, labels = plt.xticks()
    for label in labels:
        text = label.get_text().replace("−", "-").strip()
        if text:  
            label.set_text(str(int(float(text))))
    plt.xticks(range(n_int))
    plt.tight_layout()
    if title:
        suptitle = plt.title(title)
        fdct = {'color': 'black', 'fontsize': title_fontsize}
        suptitle.set(**fdct)
    if save_path:
        ax.figure.savefig(save_path + "_" + policy + ".pdf", bbox_inches='tight')
    plt.show()



def plot_model_vs_model_scores(results, dl_label = "test", score_labels = ["CTL", "ICL", "ois"],
                               checkpoint_names = None, y_lim = 1., legend_pos = "auto", 
                               title = None, title_fontsize = 16,
                               save_path = None, relabel_score = {"ois": "OIS"}):
    scores_dict = extract_scores_from_results(results, test_dl_label = dl_label, 
                                            score_labels = score_labels, 
                                            checkpoint_names = checkpoint_names)
    if relabel_score:
        for subdict in scores_dict.values():
            for old_label, new_label in relabel_score.items():
                subdict[new_label] = subdict.pop(old_label)
        score_labels = [relabel_score[old_label] if old_label in relabel_score 
                        else old_label 
                        for old_label in score_labels]
        
    checkpoint_names = list(scores_dict.keys())
    df_list = []
    for score_label in score_labels:
        df_temp = pd.DataFrame([])
        list_model_names = []
        list_score_labels = []
        list_scores = []
        for model_name in checkpoint_names:
            scores = scores_dict[model_name][score_label]
            if len(scores.reshape(-1,1)) == 1:
                scores = [scores.item()]
            else:
                scores = list(scores)
            list_scores += scores
            list_model_names += [model_name]*len(scores)
            list_score_labels += [score_label]*len(scores)
        df_temp["Model"] = list_model_names  
        df_temp["Score"] = list_score_labels
        df_temp["Value"] = list_scores
        df_list.append(df_temp)
    df = pd.concat(df_list)
    sns.set_context("paper", font_scale=1.7, rc={"font.size":1,"axes.titlesize":1,"axes.labelsize":26, 
                            "legend.fontsize":14, "legend.title_fontsize":15})
    ax = sns.catplot(x="Score", y="Value", hue="Model", kind="bar", data=df,
                    errorbar=("ci", 95), capsize=0.07, errwidth=1.5)
    ax.set_axis_labels(x_var="", y_var="")  
    if legend_pos != "auto":
        sns.move_legend(ax, "lower left", bbox_to_anchor=(legend_pos[0], legend_pos[1]), title='Model')
    if y_lim is not None:
        ax.set(ylim=(0, y_lim))
    if title:
        suptitle = plt.title(title, y = 1.0)
        fdct = {'color': 'black', 'fontsize': title_fontsize}
        suptitle.set(**fdct)
    if save_path:
        ax.figure.savefig(save_path + ".pdf", bbox_inches='tight')
    plt.show()



def plot_hard_scores(results, score_labels = ["CTL", "ICL", "ois"],
                       y_lim = 1., legend_pos = "auto", 
                       title = None, title_fontsize = 16, aspect_ratio = 1.5,
                       save_path = None, relabel_score = {"ois": "OIS"}):
    scores_dict = {dataset_label: 
               {score_label: results[dataset_label][score_label] for score_label in score_labels}
              for dataset_label in results.keys()}
    if relabel_score:
        for subdict in scores_dict.values():
            for old_label, new_label in relabel_score.items():
                subdict[new_label] = subdict.pop(old_label)
        score_labels = [relabel_score[old_label] if old_label in relabel_score 
                        else old_label 
                        for old_label in score_labels]
        
    dataset_names = list(scores_dict.keys())
    df_list = []
    for score_label in score_labels:
        df_temp = pd.DataFrame([])
        list_dataset_names = []
        list_score_labels = []
        list_scores = []
        for dataset_name in dataset_names:
            scores = scores_dict[dataset_name][score_label]
            if len(scores.reshape(-1,1)) == 1:
                scores = [scores.item()]
            else:
                scores = list(scores)
            list_scores += scores
            list_dataset_names += [dataset_name]*len(scores)
            list_score_labels += [score_label]*len(scores)
        df_temp["Dataset"] = list_dataset_names  
        df_temp["Score"] = list_score_labels
        df_temp["Value"] = list_scores
        df_list.append(df_temp)
    df = pd.concat(df_list)
    sns.set_context("paper", font_scale=1.7, rc={"font.size":1,"axes.titlesize":1,"axes.labelsize":20, 
                            "legend.fontsize":14, "legend.title_fontsize":16})
    ax = sns.catplot(x="Score",y = "Value", hue = 'Dataset', kind='bar', data=df,
                    errorbar=("ci", 95), capsize=0.07, errwidth=1.5,
                    aspect=aspect_ratio)  
    ax.set_axis_labels(x_var="", y_var="") 
    if legend_pos != "auto":
        sns.move_legend(ax, "lower left", bbox_to_anchor=(legend_pos[0], legend_pos[1]), title='Model')
    if y_lim is not None:
        ax.set(ylim=(0, y_lim))
    if title:
        suptitle = plt.title(title)
        fdct = {'color': 'black', 'fontsize': title_fontsize}
        suptitle.set(**fdct)
    if save_path:
        ax.figure.savefig(save_path + ".pdf", bbox_inches='tight')
    plt.show()



def plot_CEM_scores(results, dl_label = "test", score_labels = ["y_accuracy", "CT_MI_mix"],
                               checkpoint_names = None, y_lim = 1., legend_pos = "auto", 
                               title = None, title_fontsize = 24, axis_label_size = 20,
                               save_path = None, relabel_score = {"y_accuracy": r"$y_{acc}$",
                                                                  "CT_MI_mix": r"$\widetilde{\,I\,}^{\,(CT)}$", 
                                                                  "c_accuracy": r"$c_{acc}$",
                                                                  "avg_self_MI_mix_c_gt": r"$\widetilde{\,I\,}^{\text{(self)}}$",
                                                                  "avg_other_MI_mix_c_gt": r"$\widetilde{\,I\,}^{(IC)}$",
                                                                  "ois": "OIS",
                                                                 }):
    scores_dict = extract_scores_from_results(results, test_dl_label = dl_label, 
                                            score_labels = score_labels, 
                                            checkpoint_names = checkpoint_names)
    if relabel_score:
        for subdict in scores_dict.values():
            for old_label, new_label in relabel_score.items():
                if old_label in subdict:
                    subdict[new_label] = subdict.pop(old_label)
        score_labels = [relabel_score[old_label] if old_label in relabel_score 
                        else old_label 
                        for old_label in score_labels]
        
    checkpoint_names = list(scores_dict.keys())
    df_list = []
    for score_label in score_labels:
        df_temp = pd.DataFrame([])
        list_model_names = []
        list_score_labels = []
        list_scores = []
        for model_name in checkpoint_names:
            if len(score_labels)>1:
                scores = scores_dict[model_name][score_label]
            else:
                scores = scores_dict[model_name]
            if len(scores.reshape(-1,1)) == 1:
                scores = [scores.item()]
            else:
                scores = list(scores)
            list_scores += scores
            list_model_names += [model_name]*len(scores)
            list_score_labels += [score_label]*len(scores)
        df_temp["Model"] = list_model_names  
        df_temp["Score"] = list_score_labels
        df_temp["Value"] = list_scores
        df_list.append(df_temp)
    df = pd.concat(df_list)
    sns.set_context("paper", font_scale=1.3, rc={"font.size":1,"axes.titlesize":1,"axes.labelsize":20, 
                            "legend.fontsize":18, "legend.title_fontsize":18})
    ax = sns.catplot(x="Score",y = "Value", hue = 'Model', kind='bar', data=df,
                    errorbar=("ci", 95), capsize=0.07, errwidth=1.5) 
    ax.set_axis_labels("", "")
    for axis in ax.axes.flat:
        plt.setp(axis.get_xticklabels(), fontsize=axis_label_size)
    if legend_pos != "auto":
        sns.move_legend(ax, "lower left", bbox_to_anchor=(legend_pos[0], legend_pos[1]), title='')
    if y_lim is not None:
        ax.set(ylim=(0, y_lim))
    if title:
        suptitle = plt.title(title, y = 1.07)
        fdct = {'color': 'black', 'fontsize': title_fontsize}
        suptitle.set(**fdct)
    if save_path:
        ax.figure.savefig(save_path + ".pdf", bbox_inches='tight')
    plt.show()



def plot_alignment_leakage(results, checkpoint_names = None, 
                           obs_label = "CT_MI_alignment", 
                           dl_label = 'test', error_bars = True, confidence_level = 1.96, 
                           y_lim = [0., 1.], color = "dodgerblue", 
                           axis_label_size = 14, title_fontsize = 20, aspect_ratio = 1,
                           save_path = None, title = None,
                           yaxis_title = r"$\widetilde{\,I\,}^{\text{(align)}}$",):  
    extra_vector_scores = ("MI_pos_c", "MI_neg_c")
    scores_dict = extract_scores_from_results(results, test_dl_label = dl_label, 
                                             score_labels = [obs_label], checkpoint_names = checkpoint_names)
    plt.figure(figsize=(6, 6*aspect_ratio))
    checkpoint_names = list(scores_dict.keys())
    if ("_i" not in obs_label) and (obs_label not in extra_vector_scores):
        scores_dict_df = {k: [v.mean(), v.std()/np.sqrt(len(v)-1)] for k, v in scores_dict.items()}
        df = pd.DataFrame(scores_dict_df).T
        df = df.rename(columns={0: obs_label, 1: obs_label + "_SE"})
        df[obs_label + "_SE"] = confidence_level * df[obs_label + "_SE"]
        df.index.name = "model"
        df = df.reset_index()   
        ax = sns.barplot(df, x="model", y=obs_label, color=color)
        if error_bars:
            ax.errorbar(data=df, x="model", y=obs_label,  yerr=obs_label + "_SE", ls='', color='black', capsize=4)
        ax.set(xlabel='', ylabel=yaxis_title)
    ax.set(ylim=y_lim)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=axis_label_size)  
    if title:
        suptitle = plt.title(title)
        fdct = {'color': 'black', 'fontsize': title_fontsize}
        suptitle.set(**fdct)
    if save_path:
        ax.figure.savefig(save_path + "_" + obs_label + ".pdf", bbox_inches='tight')
    plt.show()



def plot_leakage_scores(results, checkpoint_names = None, 
                        obs_label = "ICL", selected_concepts = None,
                        dl_label = 'test', error_bars = True, confidence_level = 1.96, 
                        y_lim = [0., 1.], color = "dodgerblue", save_path = None):  
    '''
    Basic function to plot leakage scores.
    '''
    extra_vector_scores = ("MI_pos_c", "MI_neg_c")
    scores_dict = extract_scores_from_results(results, test_dl_label = dl_label, 
                                             score_labels = [obs_label], checkpoint_names = checkpoint_names)
    checkpoint_names = list(scores_dict.keys())
    if ("_i" not in obs_label) and (obs_label not in extra_vector_scores):
        scores_dict_df = {k: [v.mean(), v.std()/np.sqrt(len(v)-1)] for k, v in scores_dict.items()}
        df = pd.DataFrame(scores_dict_df).T
        df = df.rename(columns={0: obs_label, 1: obs_label + "_SE"})
        df[obs_label + "_SE"] = confidence_level * df[obs_label + "_SE"]
        df.index.name = "model"
        df = df.reset_index()   
        ax = sns.barplot(df, x="model", y=obs_label, color=color)
        if error_bars:
            ax.errorbar(data=df, x="model", y=obs_label,  yerr=obs_label + "_SE", ls='', color='black', capsize=4)
        ax.set(title=None, xlabel='Model', ylabel=obs_label)
        plt.xticks(rotation=90)
    else:
        score_reps = np.array(list(scores_dict.values()))
        if selected_concepts is None:
            selected_concepts = np.arange(score_reps.shape[2])
        df_list = []
        for i_c in selected_concepts:
            df_temp = pd.DataFrame([])
            list_model_names = []
            list_cs = []
            for model_name in checkpoint_names:
                list_model_names += [model_name]*len(score_reps[0])
                list_cs += ["c_" + str(i_c)]*len(score_reps[0])
            df_temp["Model"] = list_model_names  
            df_temp["Concept"] = list_cs
            df_temp[obs_label] = np.hstack(score_reps[:, :, i_c])
            df_list.append(df_temp) 
        df = pd.concat(df_list)
        ax = sns.catplot(x='Model',y = obs_label, hue = 'Concept', kind='bar', data=df,
                    errorbar=('ci', 95), capsize=0.15)  
        plt.xticks(rotation=90)
    ax.set(ylim=y_lim)
    if save_path:
        ax.figure.savefig(save_path + "_" + obs_label + ".pdf", bbox_inches='tight')
    plt.show()




##################################################################################################
### PCA transform of concept vectors and plotting: 
##################################################################################################
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer

def ravel_c_vec(c_vec):
    if isinstance(c_vec, torch.Tensor):
        c_in = c_vec.numpy()
    else:
        c_in = c_vec
    n_samples, n_concepts, emb_size = c_in.shape
    c_ravel = np.zeros((n_samples * n_concepts, emb_size))
    for i_c in range(n_concepts):
        c_ravel[ i_c * n_samples : (i_c + 1) * n_samples] = c_in[:, i_c]
    return c_ravel



def unravel_c_vec(c_vec, n_samples, n_concepts):
    return np.swapaxes(np.array([c_vec[ i_c * n_samples : (i_c + 1) * n_samples ] 
                                  for i_c in range(n_concepts)]), 
                        0, 1)
    


def fit_PCA_c_pos_neg(c_pos, c_neg, pca_dim = 2, 
                      return_pos_neg_transformed = True):
    n_samples, n_concepts, emb_size = c_pos.shape
    c_p = ravel_c_vec(c_pos)
    c_n = ravel_c_vec(c_neg)
    c_pn = np.vstack([c_p, c_n])
    
    normalizing = Normalizer()
    normalizing.fit(c_pn)
    normalized_data = normalizing.transform(c_pn)

    scaling = StandardScaler()
    scaling.fit(normalized_data)
    scaled_data=scaling.transform(normalized_data)

    principal=PCA(n_components=pca_dim)
    principal.fit(scaled_data)
    c_pca=principal.transform(scaled_data)
    
    def pca_transform(c_vec):
        c_ravel = ravel_c_vec(c_vec)
        c_transformed = principal.transform(
            scaling.transform(
                normalizing.transform(
                    c_ravel
                )
            )
        )
        return unravel_c_vec(c_transformed, n_samples, n_concepts)
    
    if return_pos_neg_transformed:
        c_pos_pca = unravel_c_vec(c_pca[:n_samples*n_concepts], n_samples, n_concepts)
        c_neg_pca = unravel_c_vec(c_pca[n_samples*n_concepts:], n_samples, n_concepts)
        return pca_transform, c_pos_pca, c_neg_pca
    else:
        return pca_transform



def fit_PCA_c_mix(c_sem, n_concepts, pca_dim = 2, 
                    return_mix_transformed = True):
    n_samples = c_sem.shape[0]
    c_mix = torch.reshape(c_sem, (n_samples, n_concepts, -1))
    c_mix = ravel_c_vec(c_mix)
    
    normalizing = Normalizer()
    normalizing.fit(c_mix)
    normalized_data = normalizing.transform(c_mix)

    scaling = StandardScaler()
    scaling.fit(normalized_data)
    scaled_data=scaling.transform(normalized_data)

    principal=PCA(n_components=pca_dim)
    principal.fit(scaled_data)
    c_pca=principal.transform(scaled_data)
    
    def pca_transform(c_vec):
        c_ravel = ravel_c_vec(c_vec)
        c_transformed = principal.transform(
            scaling.transform(
                normalizing.transform(
                    c_ravel
                )
            )
        )
        return unravel_c_vec(c_transformed, n_samples, n_concepts)
    
    if return_mix_transformed:
        c_mix_pca = unravel_c_vec(c_pca, n_samples, n_concepts)
        return pca_transform, c_mix_pca
    else:
        return pca_transform



def plot_pca_based_on_y_sns(c_mix_pca, y_true, i_c = 0, legend_pos = "auto", aspect = 1.1, 
                            title = None, title_fontsize = 26, font_scale=1.8, save_path = None):
    df = pd.DataFrame()
    df["pca_x"] = c_mix_pca[:, i_c, 0]
    df["pca_y"] = c_mix_pca[:, i_c, 1]
    df["y"] = y_true
    palette = sns.color_palette(None, len(y_labels)) 
    
    f = plt.figure(figsize=(6*aspect,6))
    sns.set_context("paper", font_scale = font_scale)
    ax = sns.scatterplot(data=df, x="pca_x", y="pca_y", hue="y", edgecolor="none", alpha = 1, palette = palette, 
                            )
    if legend_pos != "auto":
        sns.move_legend(ax, "lower left", bbox_to_anchor=(legend_pos[0], legend_pos[1]), title='y')
    if title:
        suptitle = plt.title(title)
        fdct = {'color': 'black', 'fontsize': title_fontsize}
        suptitle.set(**fdct)
    if save_path:
        f.savefig(save_path + ".pdf", bbox_inches='tight')
    plt.show()



def plot_pca_based_on_other_cs(c_mix_pca, y_true, c_true, i_c, i_other_cs = None, legend_size = 14, title_fontsize = 20,
                              title = None, font_scale=1.8, save_path = None,
                              xy_range = "auto", selected_y = None):    
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.keys())
    
    if not i_other_cs:
        i_other_cs = [0, 1, 2]
        i_other_cs.remove(i_c)
    c_tr_other_1 = (c_true[:, i_other_cs[0]]>0.5).numpy()
    c_tr_other_2 = (c_true[:, i_other_cs[1]]>0.5).numpy()
    if selected_y is not None:
        y_selector = np.isclose(y_true,selected_y)
    else:
        y_selector = np.ones_like(y_true)
    
    fig = plt.figure()

    mask = c_tr_other_1*c_tr_other_2*y_selector
    x = c_mix_pca[:, i_c, 0][mask]
    y = c_mix_pca[:, i_c, 1][mask]
    plt.plot(x, y, color=colors[0],
             marker='o',  markersize=4, linewidth=0, 
             label = fr"$c_{i_other_cs[0]+1}=1, c_{i_other_cs[1]+1}=1$")

    mask = (~c_tr_other_1)*c_tr_other_2*y_selector
    x = c_mix_pca[:, i_c, 0][mask]
    y = c_mix_pca[:, i_c, 1][mask]
    plt.plot(x, y, color=colors[1],
             marker='o',  markersize=4, linewidth=0, 
             label = fr"$c_{i_other_cs[0]+1}=0, c_{i_other_cs[1]+1}=1$")

    mask = c_tr_other_1*(~c_tr_other_2)*y_selector
    x = c_mix_pca[:, i_c, 0][mask]
    y = c_mix_pca[:, i_c, 1][mask]
    plt.plot(x, y, color=colors[2],
             marker='o',  markersize=4, linewidth=0, 
             label = fr"$c_{i_other_cs[0]+1}=1, c_{i_other_cs[1]+1}=0$")

    mask = (~c_tr_other_1)*(~c_tr_other_2)*y_selector
    x = c_mix_pca[:, i_c, 0][mask]
    y = c_mix_pca[:, i_c, 1][mask]
    sns.set_context("paper", font_scale = font_scale)
    plt.plot(x, y, color=colors[3], 
             marker='o',  markersize=4, linewidth=0, 
             label = fr"$c_{i_other_cs[0]+1}=0, c_{i_other_cs[1]+1}=0$")
    if title:
        suptitle = plt.title(title)
        fdct = {'color': 'black', 'fontsize': title_fontsize}
        suptitle.set(**fdct)
    plt.legend(fontsize = legend_size)
    if xy_range != "auto":
        plt.xlim(*xy_range[0])
        plt.ylim(*xy_range[1])
    if save_path:
        fig.savefig(save_path + ".pdf", bbox_inches='tight')
    plt.show()





####################################################################################################################
### Plot ground-truth interconcept MIs:   (Adapted from 2301.10367)
####################################################################################################################

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Modified from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.set_ylabel(cbarlabel, rotation=0, va="center", fontsize=25)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=25)
    ax.set_yticklabels(row_labels, fontsize=25)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=0,
        ha="right",
        rotation_mode="anchor",
    )

    # Turn spines off and create white grid.
    ax.grid(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    
    Modified from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) <= threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), fontsize=15, **kw)
            texts.append(text)

    return texts



def heatmap_groundtruth_interconcept_nMI(IC_nMI, title, textcolors=("black", "white"), save_path = None):
    '''
    Plot the interconcept mutual information matrix IC_nMI.
    '''
    fig, ax = plt.subplots(1, figsize=(8, 6))
    im, cbar = heatmap(
        np.abs(IC_nMI),
        [f"$c_{i}$" for i in range(IC_nMI.shape[0])],
        [f"$c_{i}$" for i in range(IC_nMI.shape[1])],
        ax=ax,
        cbarlabel=r" $\widetilde{\,I\,}^{(IC)}$",
        vmin=0,
        vmax=0.5,
    )
    texts = annotate_heatmap(im, valfmt="{x:.2f}", textcolors = textcolors)
    fig.tight_layout()

    fig.suptitle(title, fontsize=30)
    fig.subplots_adjust(top=0.82)
    plt.show()
    if save_path:
        fig.savefig(save_path + ".pdf", bbox_inches='tight')


