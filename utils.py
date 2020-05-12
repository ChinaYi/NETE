from matplotlib import pyplot as plt
from matplotlib import *


def segment_bars(save_path, *labels):
    num_pics = len(labels)
    color_map = plt.cm.tab10
    fig = plt.figure(figsize=(15, num_pics * 1.5))

    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=10)
    
    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1,  i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow([label], **barprops)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def segment_bars_with_confidence_score(save_path, confidence_score, labels=[]):
    num_pics = len(labels)
    color_map = plt.cm.tab10

#     axprops = dict(xticks=[], yticks=[0,0.5,1], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=15)
    fig = plt.figure(figsize=(15, (num_pics+1) * 1.5))

    interval = 1 / (num_pics+2)
    axes = []
    for i, label in enumerate(labels):
        i = i + 1
        axes.append(fig.add_axes([0.1, 1-i*interval, 0.8, interval - interval/num_pics]))
#         ax1.imshow([label], **barprops)
    titles = ['Ground Truth','Causal-TCN', 'Causal-TCN + PKI', 'Causal-TCN + MS-GRU']
    for i, label in enumerate(labels):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].imshow([label], **barprops)
#         axes[i].set_title(titles[i])
    
    ax99 = fig.add_axes([0.1, 0.05, 0.8, interval - interval/num_pics])
#     ax99.set_xlim(-len(confidence_score)/15, len(confidence_score) + len(confidence_score)/15)
    ax99.set_xlim(0, len(confidence_score))
    ax99.set_ylim(-0.2, 1.2)
    ax99.set_yticks([0,0.5,1])
    ax99.set_xticks([])
 
     
    ax99.plot(range(len(confidence_score)), confidence_score)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
    
def PKI(confidence_seq, prediction_seq, transition_prior_matrix, alpha, beta, gamma): # fix the predictions that do not meet priors
    initital_phase = 0
    previous_phase = 0
    alpha_count = 0
    assert len(confidence_seq) == len(prediction_seq)
    refined_seq = []
    for i, prediction in enumerate(prediction_seq):
        if prediction == initital_phase:
            alpha_count = 0
            refined_seq.append(initital_phase)
        else:
            if prediction != previous_phase or confidence_seq[i] <= beta:
                alpha_count = 0
            
            if confidence_seq[i] >= beta:
                alpha_count += 1
            
            if transition_prior_matrix[initital_phase][prediction] == 1:
                refined_seq.append(prediction)
            else:
                refined_seq.append(initital_phase)
            
            if alpha_count >= alpha and transition_prior_matrix[initital_phase][prediction] == 1:
                initital_phase = prediction
                alpha_count = 0
                
            if alpha_count >= gamma:
                initital_phase = prediction
                alpha_count = 0
        previous_phase = prediction

    
    assert len(refined_seq) == len(prediction_seq)
    return refined_seq