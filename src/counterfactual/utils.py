import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.visualize import *

# Define threshold on competency
COMP_THRESH = 0.9

# Create list of available image modifications
mods = ['none', 'spatial', 'brightness', 'contrast', 'saturation', 'noise', 'pixelate']

# Create list of available methods
methods = ['orig', 'igd', 'fgd', 'reco', 'lgd', 'lnn']

# Convert image modifications to integer array
def mod_to_int(mods_array):
    mods_map = {mod: idx for idx, mod in enumerate(mods)}
    idxs_array = np.array([mods_map[element] for element in mods_array.flatten()])
    return idxs_array

# Convert integer array to image modifications
def int_to_mod(idxs_array):
    mods_map = {idx: mod for idx, mod in enumerate(mods)}
    mods_array = np.array([mods_map[element] for element in idxs_array.flatten()])
    return mods_array

# Save original and generated images
def save_images(orig, gen, lbl, file):
    fig = plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.title('Original Image: {} ({})'.format(orig['prediction'], round(orig['competency'], 2)))
    visualize_img(orig['image'])

    plt.subplot(2, 2, 2)
    plt.title('Generated Image: {} ({})'.format(gen['prediction'], round(gen['competency'], 2)))
    visualize_img(gen['image'])

    plt.subplot(2, 2, 3)
    plt.title('Absolute Difference Between Images')
    visualize_img(torch.abs(gen['image'] - orig['image']))

    plt.subplot(2, 2, 4)
    plt.title('Difference Averaged over Channels')
    visualize_img(torch.mean(torch.abs(gen['image'] - orig['image']), dim=0))

    plt.suptitle('True label: {}'.format(lbl))
    plt.savefig(file)
    plt.close()

# Plot reconstruction loss distribution(s)
def plot_distributions(losses, labels, file):
    fig, ax = plt.subplots()
    for loss, label in zip(losses, labels):
        sns.kdeplot(data=loss, ax=ax, label=label)
    ax.legend()
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Probability Density')
    plt.title('Reconstruction Loss Distributions')
    plt.savefig(file)
    plt.close()

# Plot probabilistic competency estimates
def plot_competency(scores, labels, file):
    filtered_scores = []
    for these_scores in scores:
        filtered_scores.append(these_scores[~np.isnan(these_scores)])
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_scores, ax=ax)
    ax.set_xticklabels([lbl.title() for lbl in labels])
    plt.ylim(-0.05, 1.05)
    plt.title('Probabilistic Competency Estimates')
    plt.savefig(file)
    plt.close()