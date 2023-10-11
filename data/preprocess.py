import sys
# python preprocess.py D_W_15K
task = sys.argv[1]
ent_mapper = dict()
rel_mapper = dict()
ent_id = 0
rel_id = 0

triples_1 = list()
for line in open(f'{task}/rel_triples_1','r'):
    head, r, tail = [item for item in line.split()]
    if head not in ent_mapper:
        ent_mapper[head] = ent_id
        ent_id += 1
    if r not in rel_mapper:
        rel_mapper[r] = rel_id
        rel_id += 1
    if tail not in ent_mapper:
        ent_mapper[tail] = ent_id
        ent_id += 1
    triples_1.append((ent_mapper[head], rel_mapper[r], ent_mapper[tail]))

triples_2 = list()
for line in open(f'{task}/rel_triples_2','r'):
    head, r, tail = [item for item in line.split()]
    if head not in ent_mapper:
        ent_mapper[head] = ent_id
        ent_id += 1
    if r not in rel_mapper:
        rel_mapper[r] = rel_id
        rel_id += 1
    if tail not in ent_mapper:
        ent_mapper[tail] = ent_id
        ent_id += 1
    triples_2.append((ent_mapper[head], rel_mapper[r], ent_mapper[tail]))

with open(f'{task}/mapped_triples_1', 'w', encoding='utf-8') as f:
    for t in triples_1:
        f.write(f"{t[0]} {t[1]} {t[2]}\n")

with open(f'{task}/mapped_triples_2', 'w', encoding='utf-8') as f:
    for t in triples_2:
        f.write(f"{t[0]} {t[1]} {t[2]}\n")

for fold in [1,2,3,4,5]:
    for type in ["train", "valid", "test"]:
        link = list()
        for line in open(f'{task}/721_5fold/{fold}/{type}_links','r'):
            s, t = [item for item in line.split()]
            link.append((ent_mapper[s], ent_mapper[t]))
        with open(f'{task}/721_5fold/{fold}/mapped_{type}_links', 'w', encoding='utf-8') as f:
            for l in link:
                f.write(f"{l[0]} {l[1]}\n")
