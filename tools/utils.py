import glob
import os
import re
import multiprocessing as mp
from multiprocessing import Process, Queue, cpu_count, Pool
from konlpy.tag import Mecab
from typing import List, Callable
from subprocess import PIPE, run

# test
mecab = Mecab()
mecab.morphs("안녕")


def callback(x):
    print ('{} running callback with arg {}'.format(mp.current_process().name, x))


def preprocess_mecab_pool_line(params_index):
    params = params_index['params']
    idx = params_index['idx']

    file_line = params['inputs'][idx].split('-')
    file_idx = int(file_line[0])
    line_idx = int(file_line[1])
    read_from = params['files'][file_idx]

    succ = dict()
    fail = set()

    try:
        res = mecab.morphs(read_from[line_idx])
        succ.update({file_idx: {line_idx: res}})
    except Exception:
        fail.add("{}".format(file_idx))

    return (succ, fail)


def preprocess_mecab_pool(params_index):
    params = params_index['params']
    idx = params_index['idx']

    succ = set()
    fail = set()
    mecab = Mecab()

    for file_idx in range(idx, idx+1):
        try:
            read_from = open(params['inputs'][file_idx], "r").read().split('\n')
            output = ["\n".join(mecab.morphs(i)) for i in read_from]
            write_to = open(params['targets'][file_idx], "w").write("\n".join(output))
            succ.add(file_idx)
        except Exception:
            fail.add(file_idx)

    return (succ, fail)


def preprocess_shuf_pool(params_index):
    params = params_index['params']
    idx = params_index['idx']

    succ = set()
    fail = set()

    # file
    for file_idx in range(idx, idx+1):
        filename = params['inputs'][file_idx]
        try:
            res_tot_line = run(" ".join(['wc', '-l', filename]), shell=True,  stdout=PIPE, stderr=PIPE, universal_newlines=True)
            tot_line = int(re.findall('\d+', res_tot_line.stdout)[0])
            sample_line = int(tot_line * float(params['sample_rate']))
            print("{}: {}/{}".format(file_idx, sample_line, tot_line))
            res = run(['shuf', '-n', str(sample_line), filename, '-o', params['targets'][file_idx]], stdout=PIPE, stderr=PIPE, universal_newlines=True)
            succ.add(file_idx)
        except Exception:
            fail.add(file_idx)

    return (succ, fail)


def multiprocessing_with_async(params: dict, func: Callable[..., int], *args, **kwargs):
    pool = Pool(processes=int(cpu_count()/2), maxtasksperchild=int(cpu_count()/4))

    results = []
    outputs = []
    succ = set()
    fail = set()

    for i in range(len(params['inputs'])):
        params_index = {'params': params, 'idx': i}
        res = pool.apply_async(func, (params_index,), callback=callback)
        results.append(res)

    for idx, res in enumerate(results):
        try:
            outputs.append(res.get(timeout=60 * 10))
        except mp.TimeoutError:
            print('Failed at:', res)

    pool.close()
    pool.join()

    for (_succ, _fail) in outputs:
        succ.update(_succ)
        fail.update(_fail)

    return (succ, fail)
