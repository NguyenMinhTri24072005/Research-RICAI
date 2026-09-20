import json
nb = json.load(open('CODE/RICE_VISION_MAIN_PIPELINE.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    print(f'--- Cell {i} ({c["cell_type"]}) ---')
    print(src[:100] + '...')
