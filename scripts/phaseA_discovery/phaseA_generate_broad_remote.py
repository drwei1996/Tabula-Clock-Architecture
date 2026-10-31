"""Isolated broad pseudobulk dependency from existing processed CELLxGENE H5AD."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,'/root/TabulaCircadian/scripts')
import stage4_pseudobulk_clock as s
import numpy as np
import pandas as pd
import h5py
from scipy import sparse

out=Path('/autodl-fs/data/TabulaCircadian/processed/phaseA_dependency_v1')
out.mkdir(exist_ok=True)
assert not (out/'broad_normalized_mean.npz').exists(), 'Output already exists'
row=s.canonical_manifest_row(s.read_manifest(s.RAW_H5AD_MANIFEST))
original=pd.read_csv(s.FEATURES_PATH,sep='\t')
before=hashlib.sha256(s.FEATURES_PATH.read_bytes()).hexdigest()
with h5py.File(row['path'],'r') as h:
    meta,codes,_=s.group_metadata_h5(h,row)
    keys=['donor_id','tissue','cell_type','broad_cell_class']
    meta['source_group_index']=np.arange(len(meta))
    match=original.merge(meta[keys+['n_cells','source_group_index']],on=keys,validate='one_to_one',suffixes=('','_source'))
    assert len(match)==len(original) and (match.n_cells==match.n_cells_source).all()
    match=match[match.n_cells>=20].copy()
    source_to_output=np.full(len(meta),-1,dtype=np.int32)
    source_to_output[match.source_group_index]=np.arange(len(match))
    output_codes=source_to_output[codes]
    n,g=s.matrix_shape(h['X'])
    symbols=s.dataframe_column_values(h['var'],['feature_name','gene_name','gene_symbols'],g)
    assert symbols is not None
    genes=[s.canonical_gene_symbol(x) or str(x).strip().upper() for x in symbols]
    # Preserve first source feature for duplicate canonical symbols, matching stage4.
    seen=set(); keep=[]
    for i,gene in enumerate(genes):
        if gene not in seen: keep.append(i); seen.add(gene)
    keep=np.array(keep)
    sums=np.zeros((len(match),g),dtype=np.float64)
    mat=h['X']; assert mat.attrs['encoding-type']=='csr_matrix'
    for start in range(0,n,4000):
        end=min(n,start+4000); ptr=mat['indptr'][start:end+1]; a,b=int(ptr[0]),int(ptr[-1])
        xx=sparse.csr_matrix((mat['data'][a:b].astype(np.float64),mat['indices'][a:b],ptr-a),shape=(end-start,g))
        cc=output_codes[start:end]; valid=cc>=0
        indicator=sparse.csr_matrix((np.ones(valid.sum()),(cc[valid],np.flatnonzero(valid))),shape=(len(match),end-start))
        block=(indicator@xx).tocoo()
        sums[block.row,block.col]+=block.data
        if start%100000==0: print('rows',end,'/',n,flush=True)
    values=sums[:,keep]/match.n_cells.to_numpy()[:,None]
    genes=np.array(genes)[keep]
    errors={}; missing_original=[]
    for col in original.columns:
        if col.startswith('expr_'):
            j=np.flatnonzero(genes==col[5:])
            if len(j)==0:
                assert col[5:] not in s.CLOCK_GENES
                assert np.all(match[col].to_numpy()==0), col
                missing_original.append(col[5:]); continue
            assert len(j)==1, (col,len(j))
            errors[col]=float(np.max(np.abs(values[:,j[0]]-match[col].to_numpy())))
            np.testing.assert_allclose(values[:,j[0]],match[col],rtol=1e-10,atol=1e-10)
    np.savez_compressed(out/'broad_normalized_mean.npz',expression=values,genes=genes.astype(str),group_id=match.group_id.to_numpy(str))
    match[['group_id']+keys+['n_cells']].to_csv(out/'observation_metadata.tsv',sep='\t',index=False)
    pd.DataFrame({'gene':genes,'source_symbol':np.array(symbols)[keep],'source_column':keep}).to_csv(out/'gene_metadata.tsv',sep='\t',index=False)
    audit={'source':row['path'],'shape':[len(match),len(genes)],'source_cells':n,'source_genes':g,'source_groups':len(original),'identifier_columns':['group_id']+keys,'scale':'Arithmetic mean of processed source X; identical to original normalized_mean, not raw counts or pseudobulk CPM. No extra transform.','source_X_attributes':dict(mat.attrs),'uns_keys':list(h['uns'].keys()) if 'uns' in h else [],'all_69_original_gene_max_error':max(errors.values()),'overlapping_original_genes_verified':len(errors),'original_absent_symbols_exported_as_zero':missing_original,'original_feature_sha256':before,'metadata_match':True,'excluded_duplicate_symbols':g-len(genes),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (out/'dependency_audit.json').write_text(json.dumps(audit,indent=2,default=str))
assert hashlib.sha256(s.FEATURES_PATH.read_bytes()).hexdigest()==before
print(json.dumps(audit,default=str),flush=True)
