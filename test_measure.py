import os
import  json
from Exp_gain import *

truth = json.load(open('/users/zengwei/data/DD/ZW_JSON/Truth.json'))

query = 'DD16-1'
doc1 = 'ebola-45b78e7ce50b94276a8d46cfe23e0abbcbed606a2841d1ac6e44e263eaf94a93'
doc2 = 'ebola-0000e6cdb20573a838a34023268fe9e2c883b6bcf7538ebf68decd41b95ae747'
doc3 = 'ebola-012d04f7dc6af9d1d712df833abc67cd1395c8fe53f3fcfa7082ac4e5614eac6'
doc4 = 'ebola-0002c69c8c89c82fea43da8322333d4f78d48367cc8d8672dd8a919e8359e150'
doc5 = 'ebola-9e501dddd03039fff5c2465896d39fd6913fd8476f23416373a88bc0f32e793c'





# test_cmd ='python /users/zengwei/DD/github/trec-dd-jig/jig/jig.py'+ ' -runid testrun '+ ' -topic '+ query + ' -docs '+doc1+':833.00 '+doc2+ ':500.00 '+doc3+':123.00 '+doc4+ ':34.00 '+doc5+ ':5.00'

# os.system(test_cmd)


rank_list = [doc1, doc2, doc3, doc4, doc5]

reward = alphaDCG_reward_per_query(rank_list, truth[query])
print reward

reward = CT_get_reward(rank_list, truth[query])
print reward