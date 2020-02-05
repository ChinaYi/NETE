import os


def run_matlab():
    from pymatbridge import Matlab
    mlab = Matlab()
    mlab.start()
    results = mlab.run_code('run cholec80/eva/Main.m')
    meanJacc, stdJacc, meanAcc, stdAcc = mlab.get_variable('meanJacc'), mlab.get_variable(
        'stdJacc'), mlab.get_variable('meanAcc'), mlab.get_variable('stdAcc')
    mlab.stop()
    return meanJacc, stdJacc, meanAcc, stdAcc

if __name__ == '__main__':
    sample_rates = [1,2,3,4,5,6,7,8,9,10]
    accs = []
    for sample_rate in sample_rates:
        cmd1 = 'python main.py --action=train --sample_rate={}'.format(sample_rate)
        cmd2 = 'python main.py --action=predict --sample_rate={}'.format(sample_rate)
        os.system(cmd1)
        os.system(cmd2)
        accs.append(run_matlab())
        print(accs)
