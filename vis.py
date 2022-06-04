import matplotlib.pyplot as plt
import torch
import os

basedir = 'exps'
std_list = ['0.2', '0.4', '0.6', '0.8', '1.0']
lr_list = ['0.01', '0.1', '1.0', '5.0', '10.0']
bs_list = ['16', '32', '64', '128', '256']
gamma_list = ['0.5', '0.6', '0.7', '0.8', '0.9']

bestres = {}
finalres = {}


for std in std_list:
    bestres[std] = {}
    finalres[std] = {}

    testacc_list = []
    for lr in lr_list:
        ptpath = os.path.join(basedir, f"std_{std}", f"lr_{lr}", 'testacc.pt')
        testacc_list.append(torch.load(ptpath))
        bestres[std][lr] = [max(testacc_list[-1]), testacc_list[-1].index(max(testacc_list[-1]))]
        finalres[std][lr] = testacc_list[-1][-1]
    plt.figure()
    for testacc in testacc_list:
        plt.plot(testacc, marker='o')
    plt.legend(lr_list)
    plt.grid()
    plt.title(f"Different Learning Rates When std={std}")
    plt.savefig(f"test_std{std}_lr.png")
    plt.clf()

    testacc_list = []
    for bs in bs_list:
        ptpath = os.path.join(basedir, f"std_{std}", f"bs_{bs}", 'testacc.pt')
        testacc_list.append(torch.load(ptpath))
        bestres[std][bs] = [max(testacc_list[-1]), testacc_list[-1].index(max(testacc_list[-1]))]
        finalres[std][bs] = testacc_list[-1][-1]

    plt.figure()
    for testacc in testacc_list:
        plt.plot(testacc, marker='o')
    plt.legend(bs_list)
    plt.grid()
    plt.title(f"Different Batch Sizes When std={std}")
    plt.savefig(f"test_std{std}_bs.png")
    plt.clf()

    testacc_list = []
    for g in gamma_list:
        ptpath = os.path.join(basedir, f"std_{std}", f"gamma_{g}", 'testacc.pt')
        testacc_list.append(torch.load(ptpath))
        bestres[std][g] = [max(testacc_list[-1]), testacc_list[-1].index(max(testacc_list[-1]))]
        finalres[std][g] = testacc_list[-1][-1]

    plt.figure()
    for testacc in testacc_list:
        plt.plot(testacc, marker='o')
    plt.legend(gamma_list)
    plt.grid()
    plt.title(f"Different Gamma When std={std}")
    plt.savefig(f"test_std{std}_gamma.png")
    plt.clf()

with open("testacc.txt", "w") as f:
    for std in std_list:
        for lr in lr_list:
            best, bestidx = bestres[std][lr]
            f.write(f'| std={std} | lr={lr} | finalacc={finalres[std][lr]} | bestacc={best} ({bestidx+1}) |\n')

        for bs in bs_list:
            best, bestidx = bestres[std][bs]
            f.write(f'| std={std} | bs={bs} | finalacc={finalres[std][bs]} | bestacc={best} ({bestidx+1}) |\n')

        for g in gamma_list:
            best, bestidx = bestres[std][g]
            f.write(f'| std={std} | gamma={g} | finalacc={finalres[std][g]} | bestacc={best} ({bestidx+1}) |\n')
