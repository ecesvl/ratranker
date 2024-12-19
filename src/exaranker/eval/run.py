from gen_monoT5out import MyGenOut
"""
Choose dataset from 'dl20','covid','nfcorpus',
                    'dbpedia,'news','robust04', 'fiqa'

and rationale from: 'explanation','factuality', 'information_density',
                    'commonsense','textual_description','rationales'
"""
model_names = ['t5large-128','t5largeexp-128' ,'all','commonsense', 'explanation', 'factuality', 'information-density', 'textual-desrc']
rationales = ['rationales', 'explanation','rationales','commonsense','explanation','factuality', 'information_density','textual_description']

for d in ['news']:
    for i in [6, 7]:
        func = MyGenOut()
        model_name = model_names[i]
        dataset_name = d
        #func.run('model name')
        func.run(model_name, dataset_name=dataset_name, rationale=rationales[i])

for d in ['covid']:
    for i in [7]:
        func = MyGenOut()
        model_name = model_names[i]
        dataset_name = d
        # func.run('model name')
        func.run(model_name, dataset_name=dataset_name, rationale=rationales[i])

for d in ['robust04']:
    for i in range(2, 8):
        func = MyGenOut()
        model_name = model_names[i]
        dataset_name = d
        # func.run('model name')
        func.run(model_name, dataset_name=dataset_name, rationale=rationales[i])