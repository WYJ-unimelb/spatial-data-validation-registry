from case_study.suburb_boundary.utils import data_loader, rule_checker

fp = "data/" + "sa-govt-dpti-sa-suburb-boundaries-aug-2018-na.json"

data = data_loader.gpd_loader(fp)
res1 = rule_checker.check_overlap(data, auto_fix=False)
res2 = rule_checker.check_overlap(data, auto_fix=True)

print(res1.equals(res2))