import papermill as pm

pids = ["pion", "electron", "muon", "ghost", "proton", "kaon"]
pid_to_num = {
    name : i for i, name in enumerate(pids)
}


for pid1 in pids:
    for pid2 in pids:
        if pid1 == pid2:
            continue
        pm.execute_notebook(
           'XGB-each-pid.ipynb',
           'XGB_pids_pairs/{}-{}.ipynb'.format(pid1, pid2),
           parameters = dict(pid_pos=pid2, pid_neg=pid1)
        )