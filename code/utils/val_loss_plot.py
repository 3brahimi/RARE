import matplotlib.pyplot as plt
import ast

eqs = [
    # 'I_8_14',
    # 'I_14_3',
    # 'I_18_12',
    # 'I_12_4',
    # 'I_32_5',
    # 'II_8_31'
    # "I_50_26"
    # "III_8_54",
    "I_6_2"
    ]
type = "clean"
for eq in eqs:
    # read lists from txt files
    with open(f'/home/qamar/workspace/crml/code/results_custom_dp_{eq}/linear/{type}/models_all/valid_losses_all_epochs.txt', 'r') as f:
        lines = f.readlines()

    val_loss = [0]*len(lines)
    for i in range(len(lines)): 
        val_loss[i] = ast.literal_eval(lines[i])

    with open(f'/home/qamar/workspace/crml/code/results_mse_dp_{eq}/linear/{type}/models_all/valid_losses_all_epochs.txt', 'r') as f:
        lines_mse = f.readlines()

    val_loss_mse = [0]*len(lines_mse)
    for i in range(len(lines_mse)): 
        val_loss_mse[i] = ast.literal_eval(lines_mse[i])
        
    # for i in range(2):
        # plt.plot(val_loss[0], label='model'+str(i))
    plt.figure()
    plt.plot(val_loss[0], label='MSE+penalty')
    plt.plot(val_loss_mse[0], label='MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Validation MSE')
    txt = eq.replace('_', '.')
    plt.title(f'Validation MSE for {txt}')
    # change y_lim to 0 to 5 # for eq I_18_12
    # plt.ylim(0, 10)
    # log scale
    # plt.yscale('log')
    plt.legend()
    # plt.savefig(f'./val_loss_plot_{eq}_log_{type}.png') 
    plt.savefig(f'./val_loss_plot_{eq}_{type}.png') 