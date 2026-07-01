from pyesgf.search import SearchConnection

conn = SearchConnection(
    'https://esgf-node.llnl.gov/esg-search',
    distrib=True
)

experiments = ['ssp126','ssp245','ssp370','ssp585']
variables = ['pr','prw','hfls','hus','wap','ua','va', 'tas']

models = {}

for exp in experiments:
    for var in variables:
        ctx = conn.new_context(
            project='CMIP6',
            experiment_id=exp,
            variable_id=var,
            variant_label='r1i1p1f1',
            frequency='day',
            facets='project,experiment_id,variable_id,source_id'
        )

        for ds in ctx.search():
            model = ds.json['source_id'][0]
            models.setdefault(model, set()).add((exp, var))

required = {(e, v) for e in experiments for v in variables}

complete_models = sorted(
    m for m, combos in models.items()
    if combos == required
)

print("Models with COMPLETE coverage:")
for m in complete_models:
    print(m)

