import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

def evaluation(args):
    '''
    We use f-score, accuracy, MSE to evaluation the performance of different models.
    Here, the best model is selected based on the averaged f-score.
    '''
    score_test = 0.0
    score_validate = 0.0
    mdx_test = 1
    mdx_validate = 1
    memo = []
    for epoch in range(1, args.n_epoch+1):
        print('='*50)
        print('Epoch: {}'.format(epoch))
        score_dict = {}

        mem_score = {'validate': [], 'test': []}

        pred_data = np.loadtxt('../nats_results/validate_pred_{}.txt'.format(epoch))
        true_data = np.loadtxt('../nats_results/validate_true_{}.txt'.format(epoch))

        avgf1 = []
        avgaccu = []
        avgmse = []

        for k in range(args.n_tasks):
            predlb = [rt for idx, rt in enumerate(pred_data[:, k].tolist()) if true_data[idx, k] != 0]
            truelb = [rt for idx, rt in enumerate(true_data[:, k].tolist()) if true_data[idx, k] != 0]
            (p1, r1, f1, _) = precision_recall_fscore_support(truelb, predlb, average='macro')
            accu = accuracy_score(truelb, predlb)
            mse = mean_squared_error(truelb, predlb)
            avgf1.append(f1)
            avgaccu.append(accu)
            avgmse.append(mse)
            print('f_score={}, Accuracy={}, MSE={}'.format(
                np.round(f1, 4), np.round(accu, 4), np.round(mse, 4)))
            mem_score['validate'].append([p1, r1, f1, accu, mse])

        avgf1 = np.average(np.array(avgf1))
        avgaccu = np.average(np.array(avgaccu))
        avgmse = np.average(np.array(avgmse))
        print('Average f_score={}, accuracy={}, MSE={}'.format(
            np.round(avgf1, 4), np.round(avgaccu, 4), np.round(avgmse, 4)
        ))

        pred_data = np.loadtxt('../nats_results/test_pred_{}.txt'.format(epoch))
        true_data = np.loadtxt('../nats_results/test_true_{}.txt'.format(epoch))

        if avgf1 > score_validate:
            score_validate = avgf1
            mdx_validate = epoch

        avgf1 = []
        avgaccu = []
        avgmse = []

        for k in range(args.n_tasks):
            predlb = [rt for idx, rt in enumerate(pred_data[:, k].tolist()) if true_data[idx, k] != 0]
            truelb = [rt for idx, rt in enumerate(true_data[:, k].tolist()) if true_data[idx, k] != 0]
            (p1, r1, f1, _) = precision_recall_fscore_support(truelb, predlb, average='macro')
            accu = accuracy_score(truelb, predlb)
            mse = mean_squared_error(truelb, predlb)
            avgf1.append(f1)
            avgaccu.append(accu)
            avgmse.append(mse)
            print('f_score={}, Accuracy={}, MSE={}'.format(
                np.round(f1, 4), np.round(accu, 4), np.round(mse, 4)))
            mem_score['test'].append([p1, r1, f1, accu, mse])

        avgf1 = np.average(np.array(avgf1))
        avgaccu = np.average(np.array(avgaccu))
        avgmse = np.average(np.array(avgmse))
        print('Average f_score={}, accuracy={}, MSE={}'.format(
            np.round(avgf1, 4), np.round(avgaccu, 4), np.round(avgmse, 4)
        ))

        if avgf1 > score_test:
            score_test = avgf1
            mdx_test = epoch

        memo.append(mem_score)

    print('='*50)
    print('Best epoch {}'.format(mdx_validate))
    print('='*50)
    print('Val')
    out1 = []
    out2 = []
    out3 = []
    for k in range(args.n_tasks):
        [p1, r1, f1, accu, mse] = memo[mdx_validate-1]['validate'][k]
        print('f_score={}, Accuracy={}, MSE={}'.format(
            np.round(f1, 4), np.round(accu, 4), np.round(mse, 4)))
        out1.append(str(np.round(f1, 4)))
        out2.append(str(np.round(accu, 4)))
        out3.append(str(np.round(mse, 4)))
    print(' '.join(out1) + ' 0 ' + ' '.join(out2) + ' 0 ' + ' '.join(out3) + ' 0')
    print('Test')
    out1 = []
    out2 = []
    out3 = []
    for k in range(args.n_tasks):
        [p1, r1, f1, accu, mse] = memo[mdx_validate-1]['test'][k]
        print('f_score={}, Accuracy={}, MSE={}'.format(
            np.round(f1, 4), np.round(accu, 4), np.round(mse, 4)))
        out1.append(str(np.round(f1, 4)))
        out2.append(str(np.round(accu, 4)))
        out3.append(str(np.round(mse, 4)))
    print(' '.join(out1) + ' 0 ' + ' '.join(out2) + ' 0 ' + ' '.join(out3) + ' 0')
    print('='*50)
    print('Max epoch {}'.format(mdx_test))
    print('='*50)
    print('Val')
    out1 = []
    out2 = []
    out3 = []
    for k in range(args.n_tasks):
        [p1, r1, f1, accu, mse] = memo[mdx_test-1]['validate'][k]
        print('f_score={}, Accuracy={}, MSE={}'.format(
            np.round(f1, 4), np.round(accu, 4), np.round(mse, 4)))
        out1.append(str(np.round(f1, 4)))
        out2.append(str(np.round(accu, 4)))
        out3.append(str(np.round(mse, 4)))
    print(' '.join(out1) + ' 0 ' + ' '.join(out2) + ' 0 ' + ' '.join(out3) + ' 0')
    print('Test')
    out1 = []
    out2 = []
    out3 = []
    for k in range(args.n_tasks):
        [p1, r1, f1, accu, mse] = memo[mdx_test-1]['test'][k]
        print('f_score={}, Accuracy={}, MSE={}'.format(
            np.round(f1, 4), np.round(accu, 4), np.round(mse, 4)))
        out1.append(str(np.round(f1, 4)))
        out2.append(str(np.round(accu, 4)))
        out3.append(str(np.round(mse, 4)))
    print(' '.join(out1) + ' 0 ' + ' '.join(out2) + ' 0 ' + ' '.join(out3) + ' 0')