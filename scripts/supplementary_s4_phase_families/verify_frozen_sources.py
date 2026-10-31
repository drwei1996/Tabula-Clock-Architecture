#!/usr/bin/env python3
"""Read-only, stdlib-only verification of all frozen input filenames and file sizes."""
import argparse,pathlib,ast
parser=argparse.ArgumentParser();parser.add_argument('--project-root',required=True);a=parser.parse_args()
root=pathlib.Path(a.project_root).resolve()
# The authoritative dictionary is loaded safely via AST literal_eval, no imports needed.
p=pathlib.Path(__file__).with_name('tabula_full_biology_v3.py')
tree=ast.parse(p.read_text(encoding='utf-8'))
for node in tree.body:
 if isinstance(node,ast.Assign):
  for target in node.targets:
   if isinstance(target,ast.Name) and target.id=='SOURCES':
    src=ast.literal_eval(node.value);break
  else:continue
  break
else:raise RuntimeError('SOURCES not found')
missing=[]
for k,rel in src.items():
 q=root/rel
 if not q.is_file():missing.append((k,str(q)))
 else:print('[FOUND]',k,q.stat().st_size, 'bytes')
if missing:
 print('\n[MISSING] These files must be present in original frozen locations:')
 for k,path in missing:print(' ',k,path)
 raise SystemExit(2)
print('\n[READY]',len(src),'frozen inputs identified; no data were changed.')
