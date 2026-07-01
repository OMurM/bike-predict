import re, json
html = open('test.html', encoding='utf-8').read()
m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
if m:
    with open('next_data.json', 'w', encoding='utf-8') as f:
        json.dump(json.loads(m.group(1)), f, indent=2, ensure_ascii=False)
else:
    print("Not found")
