import argparse
import numpy as np
import os
# import tensorflow as tf
import torch
from codebase import utils as ut
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel

def train(model, train_loader, labeled_subset, device, tqdm, writer,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', y_status='none', reinitialize=False):

    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define a custom lambda function for the scheduler
    # This function adjusts the learning rate based on the current epoch

    i = 0
    loss_vals = []
    betas = []
    timestep = []
    
    kls = []
    reconstructions = []
    record = False
    save_folder = os.path.join(os.path.dirname(__file__), 'plots')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with tqdm(total=iter_max) as pbar:
        while True:
            for batch_idx, (xu, yu) in enumerate(train_loader):
                
                i += 1 # i is num of gradient steps taken by end of loop iteration
            
                if i % 50 == 0:
 
                    betas.append(model.beta)
                    record = True
                else:
                    record = False
                if record:
                    timestep.append(i)
                optimizer.zero_grad() 
                if y_status == 'none':
                    xu = xu.to(device).reshape(xu.size(0), -1)
                    yu = yu.new(np.eye(10)[yu]).to(device).float()
                    loss, summaries = model.loss(xu)

                elif y_status == 'semisup':
                    xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
                    yu = yu.new(np.eye(10)[yu]).to(device).float()
                    # xl and yl already preprocessed
                    xl, yl = labeled_subset
                    xl = torch.bernoulli(xl)
                    loss, summaries = model.loss(xu, xl, yl)

                    # Add training accuracy computation
                    pred = model.cls(xu).argmax(1)
                    true = yu.argmax(1)
                    acc = (pred == true).float().mean()
                    summaries['class/acc'] = acc

                elif y_status == 'fullsup':
                    # Janky code: fullsup is only for SVHN
                    # xu is not bernoulli for SVHN
                    xu = xu.to(device).reshape(xu.size(0), -1)
                    yu = yu.new(np.eye(10)[yu]).to(device).float()
                    loss, summaries = model.loss(xu, yu)

                loss.backward()
                optimizer.step()


                # Feel free to modify the progress bar
                if y_status == 'none':
                    pbar.set_postfix(
                        loss='{:.2e}'.format(loss),
                        rec = '{:.2e}'.format(summaries['gen/rec']),
                        kl = '{:.2e}'.format(summaries['gen/kl_z']),
                        beta = '{:.2e}'.format(model.beta))
                        
                    if record:
                        kls.append(summaries['gen/kl_z'])
                        reconstructions.append(summaries['gen/rec'])
                        loss_vals.append(loss)
                elif y_status == 'semisup':
                    pbar.set_postfix(
                        loss='{:.2e}'.format(loss),
                        acc='{:.2e}'.format(acc))
                    if record:
                        loss_vals.append(loss)
                elif y_status == 'fullsup':
                    pbar.set_postfix(
                        loss='{:.2e}'.format(loss),
                        kl='{:.2e}'.format(summaries['gen/kl_z']))
                pbar.update(1)

                # Log summaries
                #if i % 50 == 0: ut.log_summaries(writer, summaries, i)

                if i %500 == 0:
                    max_value = 80000
                    loss_vals_cpu = [min(loss_val.cpu().item(), max_value) for loss_val in loss_vals]
                    rec_vals_cpu = [min(rec_v.item(), max_value) for rec_v in reconstructions]
                    kl_vals_cpu = [min(kl_v.item(), 5000) for kl_v in kls]
                    
                    ##########################################
                    #nelbo loss
                    plt.plot(timestep, loss_vals_cpu, linestyle='-', color='b', label='Loss Curve')
                    plt.title('Loss Over Timesteps')
                    plt.xlabel('Timestep')
                    plt.ylabel('Loss Value')
                    plt.legend()
                    plt.grid(True)
                    

                    # Save the plot as a JPG file in the 'plots' folder
                    save_path = os.path.join(save_folder, 'loss_plot.jpg')
                    plt.savefig(save_path)
                    plt.clf()

                    ###############################################
                    #Reconstruction loss plot
                    plt.plot(timestep, rec_vals_cpu, linestyle='-', color='b', label='reconstruction Curve')
                    plt.title('rec loss Over Timesteps')
                    plt.xlabel('Timestep')
                    plt.ylabel('rec Value')
                    plt.legend()
                    plt.grid(True)

                    # Save the plot as a JPG file in the 'plots' folder
                    save_path = os.path.join(save_folder, 'rec_plot.jpg')
                    plt.savefig(save_path)
                    plt.clf()

                    #####################################################
                    #KL Divergence plot
                    plt.plot(timestep, kl_vals_cpu, linestyle='-', color='b', label='KL Curve')
                    plt.title('KL Loss Over Timesteps')
                    plt.xlabel('Timestep')
                    plt.ylabel('KL Value')
                    plt.legend()
                    plt.grid(True)

                    # Save the plot as a JPG file in the 'plots' folder
                    save_path = os.path.join(save_folder, 'KL_plot.jpg')
                    plt.savefig(save_path)
                    plt.clf()


                    #####################################################
                    #Beta plot
                    plt.plot(timestep, betas, linestyle='-', color='b', label='Beta Curve')
                    plt.title('Beta Over Timesteps')
                    plt.xlabel('Timestep')
                    plt.ylabel('Beta Value')
                    plt.legend()
                    plt.grid(True)

                    # Save the plot as a JPG file in the 'plots' folder
                    save_path = os.path.join(save_folder, 'Beta_plot.jpg')
                    plt.savefig(save_path)
                    plt.clf()

                   

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i)
                
               
                periods = 6
                period_size = iter_max // periods

                if i < iter_max:
                    period_idx = i // period_size
                    within_period_idx = i % period_size

                    if within_period_idx < period_size // 2:
                        # First half of the period: Increase beta at a fast rate
                        model.beta = model.beta_start + (model.beta_end - model.beta_start) * (2 * within_period_idx / period_size)
                    else:
                        # Second half of the period: Stay at beta=1
                        model.beta = model.beta_end

                    if within_period_idx == period_size - 1 and period_idx < periods - 1:
                        # Transition to the next period: Reset beta to 0
                        model.beta = model.beta_start
                else:
                    model.beta = model.beta_end
                

                if i == iter_max:
                    return
