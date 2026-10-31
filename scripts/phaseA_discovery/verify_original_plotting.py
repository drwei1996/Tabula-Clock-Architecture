"""Compare the refactor to AST-extracted original plotting functions only.

No original module imports, top-level actions, analysis functions or main() run.
No original table/figure is written. Original plots are rendered in memory only.
"""
import ast
import json
from pathlib import Path

import revision_v2_plots as revised
from revision_v2_plots import np, pd, plt, sns
from publication_style import ROOT, LINEAGE_ORDER, CORE_CLOCK, DISPLAY_LABELS, LINEAGE_COLORS, COMPONENT_COLORS

def check_reference():
    source=revised.DEFAULT_SOURCE
    path=source/'scripts/revision_v2_runner.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    allowed={
        'plot_formal_variance':'variance', 'plot_variance_comparison':'comparison',
        'plot_corrected_robustness':'robustness', 'plot_lineage_atlas':'atlas',
        'plot_paired_map':'paired', 'plot_pathway_context':'pathway',
    }
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in allowed]
    assert len(functions)==len(allowed)
    captured=[]
    namespace=dict(pd=pd,np=np,plt=plt,sns=sns,Line2D=revised.Line2D,Path=Path,
                   LINEAGE_ORDER=LINEAGE_ORDER,LINEAGE_LABELS=DISPLAY_LABELS,LINEAGE_COLORS=LINEAGE_COLORS,
                   CORE_CLOCK=CORE_CLOCK,MIN_PAIRED_DONORS=3,COMPONENT_COLORS=COMPONENT_COLORS,
                   short_label=revised.short_label,DIRS={k:ROOT for k in ['source_data','figures_main','figures_extended','figure_prototypes']},
                   save_figure=lambda fig,*args:captured.append(fig),register_panel=lambda *args:None)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    results={}
    revised.apply_publication_style()
    for name,key in allowed.items():
        data=pd.read_csv(source/'source_data'/revised.SOURCES[key],sep='\t')
        namespace[name](data.copy(deep=True)); original=captured.pop()
        new=getattr(revised,name)(data.copy(deep=True))
        if key in ['paired','pathway']:
            old=np.ma.filled(original.axes[0].collections[0].get_array(),np.nan)
            current=np.ma.filled(new.axes[0].collections[0].get_array(),np.nan)
            np.testing.assert_allclose(old,current,rtol=0,atol=0,equal_nan=True)
            def annotations(fig):
                return [(text.get_position(),text.get_text()) for text in fig.axes[0].texts]
            assert annotations(original)==annotations(new)
            results[key]={'matrix_identical':True,'annotations_identical':True}
        elif key=='atlas':
            for old_ax,new_ax in zip(original.axes[:5],new.axes[:5]):
                old=old_ax.collections[0]; cur=new_ax.collections[0]
                np.testing.assert_array_equal(old.get_offsets(),cur.get_offsets())
                np.testing.assert_array_equal(old.get_array(),cur.get_array())
                np.testing.assert_array_equal(old.get_sizes(),cur.get_sizes())
            results[key]={'five_panel_selection_coordinates_values_and_sizes_identical':True}
        elif key=='variance':
            np.testing.assert_array_equal([p.get_width() for p in original.axes[0].patches],
                                          [p.get_width() for p in new.axes[0].patches])
            results[key]={'all_bar_widths_identical':True}
        elif key=='comparison':
            old=np.concatenate([c.get_offsets()[:,1] for c in original.axes[0].collections if len(c.get_offsets())])
            cur=np.concatenate([c.get_offsets()[:,1] for c in new.axes[0].collections if len(c.get_offsets())])
            np.testing.assert_array_equal(np.sort(old),np.sort(cur))
            results[key]={'all_estimates_identical':True,'number_of_estimates':len(old)}
        elif key=='robustness':
            old=np.concatenate([c.get_offsets() for c in original.axes[0].collections])
            cur=np.concatenate([c.get_offsets() for c in new.axes[0].collections])
            np.testing.assert_array_equal(old[np.lexsort(old.T)],cur[np.lexsort(cur.T)])
            results[key]={'all_paired_estimates_identical':True}
        plt.close(original); plt.close(new)
    # Exercise absent-estimate handling without changing or saving any source data.
    data=pd.read_csv(source/'source_data'/revised.SOURCES['robustness'],sep='\t')
    data.loc[0,'interaction_model_r2']=np.nan; data.loc[1,'main_model_r2']=np.nan
    fig=revised.plot_corrected_robustness(data)
    assert sum(len(c.get_offsets()) for c in fig.axes[0].collections)==6
    assert sum('not evaluated' in text.get_text() for text in fig.axes[1].texts)==2
    plt.close(fig)
    results['missing_estimate_regression']={'absent_estimates_not_plotted_as_zero':True,'explicit_notes':2}
    (ROOT/'figures_revised/reference_comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    check_reference()
