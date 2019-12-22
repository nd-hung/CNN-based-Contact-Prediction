import os,sys
from multiprocess import Pool

def extract_feature(target):
    ss = aln_dir + target + '.aln'
    ts = feat_dir + target + '.21c'
    os.system('bin/cov21stats ' + ss + ' ' + ts)

aln_dir = sys.argv[1]# '/data/test/psicov150/aln/'
feat_dir = sys.argv[2]# '/data/test/psicov150/21c/'
target_file = sys.argv[3]# '/data/test/psicov150/target.lst.sorted'

list_of_targets = [e.strip() for e in open(target_file) if e.strip()]
print(len(list_of_targets))

pool = Pool(12)
pool.map(extract_feature, list_of_targets)
