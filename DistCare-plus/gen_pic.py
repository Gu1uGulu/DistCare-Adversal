import matplotlib.pyplot as plt


def gen_pic(file):
    fold = 0
    epoch = 0
    mse_list = [[], [], [], [], [], [], [], [], [], []]
    for line in open(file, "r", encoding='UTF-8'):
        fold_index = line.find('Fold')
        epoch_index = line.find('Epoch')
        best_index = line.find('Save FOLD-BEST model')
        if epoch_index != -1:
            epoch += 1
        if fold_index != -1:
            if fold != 10 and int(line[fold_index + 5]) != fold:
                fold += 1
                epoch = 0
        if best_index != -1:
            mse_list[fold - 1].append((epoch, float(line[best_index + 28:best_index + 35])))
    return mse_list


if __name__ == '__main__':
    list1 = gen_pic('models_and_logs/TongJi/target_plus_150/target_plus.log')
    list2 = gen_pic('models_and_logs/TongJi/target_origin_150/target_origin.log')
    for fold in range(10):
        x_ours, y_ours = [each[0] for each in list1[fold]], [each[1] for each in list1[fold]]
        x_DC, y_DC = [each[0] for each in list2[fold]], [each[1] for each in list2[fold]]
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title('MSE')
        plt.plot(x_ours, y_ours, color='green', linestyle='-', linewidth=1, label='Ours')
        plt.plot(x_DC, y_DC, color='red', linestyle='-', linewidth=1, label='DistCare')
        plt.legend()
        plt.savefig('image/' + 'fold_' + str(fold + 1) + '.png')
        plt.clf()
