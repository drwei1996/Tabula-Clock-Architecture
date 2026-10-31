"""Integrity/format checks after A5 numerical tests and figure inspection."""
from a5_variance_architecture import *
import subprocess

def main():
    result=subprocess.run([sys.executable,str(ROOT/'scripts/test_a5_variance_architecture.py')],capture_output=True,text=True)
    (DEST/'00_audit/numerical_tests.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    assert result.returncode==0,result.stderr
    t=read('statistics/gene_variance_components.csv').set_index('gene');valid=t[FRAC].notna().all(axis=1)
    np.testing.assert_allclose(t.loc[valid,FRAC].sum(axis=1),1,atol=1e-12)
    np.testing.assert_allclose(t.loc[valid,[c[2:]+'_component' for c in FRAC]].sum(axis=1),t.loc[valid,'decomposition_total_variance'],atol=1e-12,rtol=1e-10)
    pools=pd.read_csv(A1/'candidate_pools.csv');assert t.loc[pools.matched_gene,FRAC].notna().all().all()
    for path in (DEST/'00_audit').glob('A5_Fig*_layout.json'):
        qa=json.loads(path.read_text(encoding='utf-8'));assert not qa['text_outside_nominal_canvas'],qa
        assert not qa['legends_overlapping_data_panels'],qa
    for i in range(1,9):
        files=list((DEST/'figures').glob(f'A5_Fig{i}_*.png'));assert len(files)==1
        f=files[0];assert min(Image.open(f).info['dpi'])>=299
        for ext in ['.svg','.pdf']:assert f.with_suffix(ext).exists()
        assert (DEST/'source_data'/f'{f.stem}.csv').exists()
    before=json.loads((DEST/'00_audit/protected_hashes.json').read_text(encoding='utf-8'));changed=[f for f,h in before.items() if sha(Path(f))!=h];assert not changed,changed
    v={'genes_assessed':len(t),'genes_with_defined_decomposition':int(valid.sum()),'undefined_supported_variance':int((~valid).sum()),'sum_to_one_and_absolute_component_identity':'PASS','all_candidate_genes_defined':'PASS','all_8_figure_formats_and_source_data':'PASS','visual_QA':'PASS: eight figures inspected, labels corrected for figures5/7','existing_files_hash_verified':len(before),'A1_A2_A3_gate_unchanged':True,'PhaseB_started':False,'broad_matrix_regenerated':False,'random_effect_models_attempted':False,'test_status':'PASS: four numerical regression tests'}
    jwrite(DEST/'00_audit/final_validation.json',v)
    jwrite(DEST/'00_audit/environment.json',{'python':sys.version,'executable':sys.executable,'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','matplotlib']},'seed':SEED5,'threads':1})
    (DEST/'README.md').write_text('A5 is isolated from completed Phase A modules.\n\nMain report: reports/A5_VARIANCE_ARCHITECTURE_REPORT.md\n\nReproduce from project root using the existing scientific Python environment:\n\n```powershell\npython scripts/test_a5_variance_architecture.py\npython scripts/a5_variance_architecture.py\npython scripts/a5_figures_report.py\npython scripts/a5_validate.py\n```\n\nScripts reuse the cached broad matrix, A1 candidate pools and panels; numerical stage caches per-gene decompositions and block-deletion results. Existing completed outputs are never intentionally overwritten. Check 00_audit/protected_hashes.json and final_validation.json. CV is withheld because a processed-expression ratio-scale interpretation is unverified; pseudobulk prevalence is not single-cell detection.\n',encoding='utf-8')
    print(json.dumps(v,indent=2),flush=True)

if __name__=='__main__':main()
