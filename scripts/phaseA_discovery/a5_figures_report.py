"""Figures and final report from cached A5 statistics only."""
import os
from a5_variance_architecture import *
import textwrap

def finish(fig,name,source):
    only=os.environ.get('A5_FIGURE_FILTER')
    if only and name!=only:
        plt.close(fig);return
    write(f'source_data/{name}.csv',source)
    qa=save_figure(fig,name,DEST/'figures',dpi=300)
    jwrite(DEST/f'00_audit/{name}_layout.json',qa)

def figures():
    apply_publication_style();t=read('statistics/gene_variance_components.csv').set_index('gene');clock=t[t.is_expanded];bg=t[t.eligible_background];order=[g[5:] for g in protocol()['feature_sets']['EXPANDED_FEATURE_COLS']];clock=clock.loc[order];core=set(clock[clock.is_core].index)
    pools=pd.read_csv(A1/'candidate_pools.csv').query('variant == "primary"');selected=set(pools.matched_gene)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));fig.subplots_adjust(left=.065,right=.97,bottom=.23,top=.78,wspace=.32)
    for ax,xcol,ycol in zip(axes,['mean_expression','total_variance','mean_expression'],['total_variance','lineage_eta_squared','tissue_eta_squared']):
        x=np.log10(bg[xcol]);y=np.log10(bg[ycol]) if ycol=='total_variance' else bg[ycol]
        ax.hexbin(x,y,gridsize=65,mincnt=1,bins='log',cmap='Greys',linewidths=0)
        for label,part,color,marker in [('Matched candidates',bg.loc[bg.index.isin(selected)],'#59A14F','.'),('Core clock',clock[clock.is_core],'#4E79A7','o'),('Expanded-only clock',clock[~clock.is_core],'#E15759','^')]:
            ax.scatter(np.log10(part[xcol]),np.log10(part[ycol]) if ycol=='total_variance' else part[ycol],s=4 if marker=='.' else 28,c=color,marker=marker,alpha=.18 if marker=='.' else .95,label=label,edgecolors='none')
        ax.set(xlabel='log10 '+xcol.replace('_',' '),ylabel=('log10 ' if ycol=='total_variance' else '')+ycol.replace('_',' '));clean_axis(ax)
    fig.suptitle('A5 · Clock genes in the genome-wide variance landscape',y=.98);fig.legend(handles=[Line2D([],[],marker=m,ls='',color=c,label=l,markersize=6) for l,c,m in [('Matched candidates','#59A14F','o'),('Core clock','#4E79A7','o'),('Expanded-only clock','#E15759','^')]],loc='lower center',ncol=3)
    finish(fig,'A5_Fig1_genome_variance_landscape',t.reset_index()[['gene','mean_expression','total_variance','lineage_eta_squared','tissue_eta_squared','is_core','is_expanded','selected_primary_expanded']])

    fig,ax=plt.subplots(figsize=(8,7));fig.subplots_adjust(left=.17,right=.86,bottom=.21,top=.88)
    im=ax.imshow(clock[FRAC],aspect='auto',cmap='Blues',vmin=0,vmax=max(.5,clock[FRAC].max().max()))
    ax.set_yticks(range(17),[g+(' *' if g in core else '') for g in order]);ax.set_xticks(range(5),['Lineage','Tissue','Donor','Lineage × tissue','Residual'],rotation=25,ha='right');clean_axis(ax,heatmap=True)
    for i in range(17):
        for j in range(5):ax.text(j,i,f'{clock.iloc[i][FRAC[j]]:.2f}',ha='center',va='center',fontsize=9,color='white' if clock.iloc[i][FRAC[j]]>.36 else '#222222')
    cax=fig.add_axes([.90,.26,.025,.50]);fig.colorbar(im,cax=cax,label='Fraction of supported-subset variance')
    fig.suptitle('A5 · Static-expression variance allocation',y=.97);fig.text(.17,.05,'* Core clock; all 17 genes form Expanded clock. Fixed-effect SS allocation.',fontsize=9)
    finish(fig,'A5_Fig2_clock_fraction_heatmap',clock.reset_index())

    fig,ax=plt.subplots(figsize=(12,5));fig.subplots_adjust(left=.07,right=.99,bottom=.29,top=.83);bottom=np.zeros(17)
    for m,color in zip(FRAC,COLORS):ax.bar(range(17),clock[m],bottom=bottom,color=color,label=m[2:].replace('_',' '),width=.77);bottom+=clock[m].to_numpy()
    ax.set(ylim=(0,1),ylabel='Fraction of supported-subset variance');ax.set_xticks(range(17),order,rotation=55,ha='right');clean_axis(ax);fig.suptitle('A5 · Gene-by-gene variance composition',y=.97);fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=5)
    finish(fig,'A5_Fig3_clock_stacked_composition',clock.reset_index())

    fig,axes=plt.subplots(1,2,figsize=(12,8));fig.subplots_adjust(left=.12,right=.98,bottom=.11,top=.88,wspace=.30);source=[]
    for ax,m in zip(axes,['f_lineage','context_fraction']):
        values=[]
        for g in order:
            candidates=pools[pools.clock_gene.eq(g)].matched_gene;v=t.loc[candidates,m];values.append(v.dropna().to_numpy());source.extend({'clock_gene':g,'candidate':c,'metric':m,'value':t.loc[c,m],'clock_value':t.loc[g,m]} for c in candidates)
        ax.boxplot(values,orientation='horizontal',tick_labels=order,showfliers=False,patch_artist=True,boxprops={'facecolor':'#E8E8E8'},medianprops={'color':'#555555'})
        ax.scatter(clock[m],np.arange(1,18),c=['#4E79A7' if g in core else '#E15759' for g in order],s=30,marker='D',zorder=3)
        ax.invert_yaxis();ax.set(xlabel=m.replace('_',' '),title='Lineage allocation' if m=='f_lineage' else 'Context = tissue + interaction');clean_axis(ax)
    fig.suptitle('A5 · Clock genes versus their own matched candidates',y=.97);fig.text(.5,.025,'Boxes: candidate IQR and 1.5×IQR whiskers; diamonds: clock genes. No outcome-based matching.',ha='center',fontsize=9)
    finish(fig,'A5_Fig4_matched_candidate_distributions',pd.DataFrame(source))

    null=read('statistics/panel_null_distributions.csv').query('variant == "primary"');tests=read('statistics/panel_matched_tests.csv').query('variant == "primary"')
    fig,axes=plt.subplots(2,6,figsize=(19,7.5));fig.subplots_adjust(left=.045,right=.99,bottom=.15,top=.84,hspace=.85,wspace=.40)
    for i,panel in enumerate(['core','expanded']):
        for j,m in enumerate(MET):
            ax=axes[i,j];v=null[(null.panel==panel)&(null.metric==m)].value;r=tests[(tests.panel==panel)&(tests.metric==m)].iloc[0]
            ax.hist(v,bins=25,color='#C9CDD3',edgecolor='white',linewidth=.4);ax.axvline(r.observed,color='#4E79A7' if i==0 else '#E15759',lw=2);ax.set_title(f'{panel.title()} · {m[2:] if m.startswith("f_") else "context"}\nP={r.P:.3g}; q={r.BH12:.3g}',fontsize=10);ax.set_xlabel('Panel median' if i else '');clean_axis(ax)
            if j==0:ax.set_ylabel('Null panel count')
    fig.suptitle('A5 · Prespecified panel-level matched-null comparisons',y=.98);fig.text(.5,.035,'1,000 existing panels per clock set; context upper-tail, other fractions two-sided; BH over 12 primary tests.',ha='center',fontsize=10)
    finish(fig,'A5_Fig5_panel_empirical_nulls',null.merge(tests[['panel','metric','observed','P','BH12']],on=['panel','metric']))

    projection=read('source_data/composition_projection.csv');fig,axes=plt.subplots(1,2,figsize=(13,6));fig.subplots_adjust(left=.07,right=.98,bottom=.22,top=.82,wspace=.28)
    centroidrows=[]
    for ax,method in zip(axes,['hellinger','raw_fraction']):
        d=projection[projection.method.eq(method)].set_index('gene');b=d[d.sample_background];ax.scatter(b.PC1,b.PC2,s=4,c='#BBBBBB',alpha=.22,label='Uniform background sample')
        c=d[d.is_matched_candidate];ax.scatter(c.PC1,c.PC2,s=5,c='#59A14F',alpha=.25,label='Matched candidates')
        for g in order:
            members=pools[pools.clock_gene.eq(g)].matched_gene;cent=d.loc[members,['PC1','PC2']].mean();point=d.loc[g,['PC1','PC2']]
            ax.plot([point.PC1,cent.PC1],[point.PC2,cent.PC2],color='#888888',alpha=.4,lw=.6);centroidrows.append({'gene':g,'method':method,'candidate_centroid_PC1':cent.PC1,'candidate_centroid_PC2':cent.PC2})
        c=d.loc[order];ax.scatter(c.PC1,c.PC2,s=28,c='#E15759',label='Clock genes',zorder=5)
        for g in ['NFIL3','RORA','CRY1','HLF']:ax.annotate(g,(d.loc[g,'PC1'],d.loc[g,'PC2']),xytext=(5,5),textcoords='offset points',fontsize=8)
        model=json.loads((DEST/f'statistics/{method}_PCA.json').read_text());ax.set(xlabel=f'PC1 ({model["explained_variance_ratio"][0]:.1%})',ylabel=f'PC2 ({model["explained_variance_ratio"][1]:.1%})',title='Hellinger square-root composition' if method=='hellinger' else 'Raw fractions · sensitivity');clean_axis(ax)
    fig.suptitle('A5 · Variance-composition space',y=.98);fig.legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=l,markersize=5) for l,c in [('Uniform background sample','#BBBBBB'),('Matched candidates','#59A14F'),('Clock genes','#E15759')]],loc='lower center',ncol=3);fig.text(.5,.10,'Lines connect each clock gene to its own candidate centroid; PCA fitted to eligible background.',ha='center',fontsize=9)
    finish(fig,'A5_Fig6_composition_space',projection);write('source_data/A5_Fig6_candidate_centroids.csv',pd.DataFrame(centroidrows))

    linked=read('statistics/A2_A5_gene_join.csv');corr=read('statistics/A2_A5_correlations.csv');fig,axes=plt.subplots(1,3,figsize=(14,5));fig.subplots_adjust(left=.07,right=.98,bottom=.23,top=.77,wspace=.35)
    for ax,old,new,label in zip(axes,['lineage_stability','tissue_remodeling','cell_identity_tissue_context_interaction_adj_r2'],['f_lineage','context_fraction','f_lineage_tissue'],['A2 lineage stability','A2 tissue remodeling','A2 interaction score']):
        ax.scatter(linked[old],linked[new],c=['#4E79A7' if g in core else '#E15759' for g in linked.gene],s=30)
        for g in ['NFIL3','RORA','CRY1','HLF']:
            r=linked[linked.gene.eq(g)].iloc[0]
            offset={'NFIL3':(16,20),'RORA':(8,-18),'CRY1':(-28,16),'HLF':(5,5)}[g] if new=='f_lineage_tissue' else (5,5)
            ax.annotate(g,(r[old],r[new]),xytext=offset,textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','color':'#888888','lw':.5} if new=='f_lineage_tissue' else None)
        ax.margins(x=.15,y=.15)
        r=corr[(corr.A2==old)&(corr.A5==new)].iloc[0];ax.set(xlabel=label,ylabel=new.replace('_',' '),title=f'ρ = {r.rho:.2f}\nBootstrap interval [{r.bootstrap_low:.2f}, {r.bootstrap_high:.2f}]');clean_axis(ax)
    fig.suptitle('A5 · Relationship to frozen A2 scores',y=.98);fig.text(.5,.06,'17 clock genes; same-data methodological comparison, not independent biological validation.',ha='center',fontsize=10)
    finish(fig,'A5_Fig7_A2_relationship',linked)

    robust=read('statistics/robustness_panel_tests.csv');d=robust[robust.scenario.str.startswith('delete_')].copy();d['test']=d.panel+' · '+d.metric.str.replace('f_','',regex=False);d['block']=d.scenario.str.replace('delete_','',regex=False)+': '+d.removed_or_rule.astype(str)
    matrix=d.pivot(index='test',columns='block',values='delta');fig,ax=plt.subplots(figsize=(17,7));fig.subplots_adjust(left=.22,right=.89,bottom=.43,top=.86)
    limit=np.abs(matrix.to_numpy()).max();im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit);ax.set_yticks(range(len(matrix)),matrix.index);ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=75,ha='right',fontsize=7);clean_axis(ax,heatmap=True)
    cax=fig.add_axes([.91,.46,.015,.34]);fig.colorbar(im,cax=cax,label='Clock median − matched-null median');fig.suptitle('A5 · Donor and tissue deletion sensitivity',y=.97)
    finish(fig,'A5_Fig8_robustness',d)

def report():
    t=read('statistics/gene_variance_components.csv');clock=t[t.is_expanded];bg=t[t.eligible_background];tests=read('statistics/panel_matched_tests.csv');primary=tests.query('variant == "primary"');rob=read('statistics/robustness_summary.csv');gene=read('statistics/gene_candidate_tests.csv').query('variant == "primary"');corr=read('statistics/A2_A5_correlations.csv');negative=read('statistics/negative_control_multivariable.csv');comp=read('statistics/composition_space_tests.csv');fail=read('statistics/model_failures.csv');expression=json.loads((DEST/'statistics/expression_variance_model.json').read_text()); adjusted=read('statistics/clock_expression_adjusted_variance.csv')
    sensitivity=read('statistics/robustness_panel_tests.csv');robust=[]
    for r in rob.itertuples():
        others=sensitivity[(sensitivity.panel==r.panel)&(sensitivity.metric==r.metric)&sensitivity.scenario.isin(['minimum_support','alternative_model'])]
        robust.append(r.primary_BH<.05 and r.deletion_direction_consistency>=.9 and r.deletion_BH_significant_fraction>=.8 and (np.sign(others.delta)==np.sign(r.primary_delta)).all())
    strong=rob[np.array(robust)];anyprimary=(primary.BH12<.05).any();anygene=(gene.BH85<.05).any()
    decision='A) CLOCK-SPECIFIC VARIANCE ARCHITECTURE SUPPORTED' if len(strong) and set(strong.panel)=={'core','expanded'} else 'B) PARTIAL / COMPONENT-SPECIFIC VARIANCE ARCHITECTURE' if anyprimary or anygene else 'C) HIGH VARIANCE BUT NOT CLOCK-SPECIFIC'
    # Wide-composition distance is secondary; if isolated, describe partial rather than overlook it.
    if decision.startswith('C)') and (comp.query('space == "hellinger"').BH4<.05).any():decision='B) PARTIAL / COMPONENT-SPECIFIC VARIANCE ARCHITECTURE'
    panel_summary=clock.groupby('is_core')[['total_variance']+MET].median()
    medians=pd.DataFrame([{'panel':p,'n_genes':len(d),**{m:d[m].median() for m in ['mean_expression','total_variance','mean_expression_background_percentile','total_variance_background_percentile']+MET}} for p,d in [('Core',clock[clock.is_core]),('Expanded',clock),('Eligible background',bg)]])
    write('statistics/panel_descriptive_summary.csv',medians)
    gene['absolute_delta']=gene.delta.abs();patterns=gene.sort_values(['BH85','absolute_delta'],ascending=[True,False]).head(12)
    a2sel=corr[((corr.A2=='lineage_stability')&(corr.A5=='f_lineage'))|((corr.A2=='tissue_remodeling')&(corr.A5=='context_fraction'))|((corr.A2=='cell_identity_tissue_context_interaction_adj_r2')&(corr.A5=='f_lineage_tissue'))]
    rank=clock[['gene','mean_expression_background_percentile','total_variance_background_percentile','lineage_eta_squared_background_percentile','tissue_eta_squared_background_percentile']+MET]
    n_ok=int(t[FRAC].notna().all(axis=1).sum())
    family_frame=read('statistics/exploratory_family_summary.csv') if (DEST/'statistics/exploratory_family_summary.csv').exists() else pd.DataFrame()
    text=f'''# A5 — Clock gene variance architecture

## Scope, method and interpretation
**FACT:** {len(t):,} sufficiently measured genes assessed; {n_ok:,} have defined variance fractions; {len(fail):,} have no measurable supported-subset variance and retain NA fractions. The validated broad matrix and every completed A1/A2/A3 output are reused read-only. No Phase B or gate change. Seeds and prespecification are in `00_audit/prespecified_protocol.json`; exact observations, universes, exclusion reasons, input SHA256s and candidate memberships are in `00_audit/`.

The primary method is a **finite-design hierarchical Shapley sums-of-squares decomposition**, consistently applied to all genes. It is a fixed-effect fallback, not a REML random-effect variance estimate. There are 936 supported pseudobulks, 5 lineages, 29 tissues, 18 donors, and 97 observed lineage/tissue combinations out of 145 possible. Additive model rank is 50; observed-pair-plus-donor rank is 114; residual degrees of freedom is 822. Missing combinations are not imputed. SVD explicitly handles categorical redundancy. All clock and primary matched-candidate genes must have defined fractions for panel inference; no silent dropped features.

Average lineage, tissue and donor SS increments across all six insertion orders, then add lineage×tissue SS beyond the additive model; residual is the full-model SSE. This is a hierarchical allocation of shared observed variance, not five uniquely identifiable biological causes. Absolute components sum to `decomposition_total_variance` on the 936 supported observations; `total_variance` describes all 2,030 and is intentionally a separate quantity. Fractions sum to 1. `context_fraction=f_tissue+f_lineage_tissue` is coherent under this allocation. All fractions are unadjusted finite-sample fitted SS fractions; rank/complexity effects are controlled by the identical design for clock and null, not by pretending this is unbiased population variance. Donor-first allocation is an explicit alternative sensitivity.

**Model choice:** fixed named lineages (only five groups), selected tissues, partially crossed support, and dependent fine-cell-type pseudobulks motivate a finite-design estimand. REML was not attempted; there are no claimed random-effect convergence failures. `model_failures.csv` records zero-variation cases; `design_diagnostics.json` and robustness diagnostics record ranks/support. Thus A5 does not satisfy a stronger claim of uniquely separated random-effect variance parameters. All genes use one compatible model.

Mean, population variance and tissue/donor marginal eta-squared use all 2,030 observations; lineage marginal eta-squared reuses the audited 1,302-observation calculation. Source values are arithmetic means of processed X, not raw counts or CPM; a ratio-scale interpretation is not established. Accordingly CV is withheld (NA); the arithmetic SD/mean ratio is exported separately with no biological CV interpretation. Broad true single-cell detection is unavailable: positive-pseudobulk prevalence is a clearly labeled proxy, with true detection retained only where supplied. Neither is a matching variable in the original A1 null.

## A5.1–A5.3: genome-wide context
**FACT:** Panel medians (background comparisons are descriptive, not the specificity test):

{markdown_table(medians)}

Per-clock genome-wide percentiles and fractions:

{markdown_table(rank)}

**INTERPRETATION:** Total variance alone is not specificity. An eligible-background model `log10 variance ~ log10 mean + log10 mean squared + pseudobulk prevalence` explains {expression['R2']:.1%} of log-variance variation. Clock conditional residual percentiles are below; this is a descriptive expression-regime adjustment, not a causal explanation.

{markdown_table(adjusted)}

## A5.4: prespecified matched specificity tests
All original panels and gene-specific candidate pools are reused. Primary pools contain nearest 100 nonclock genes in standardized log expression/variance; no outcome-based rematching. Core and Expanded each have 1,000 existing primary panels. Three existing sensitivities (second seed, nearest25, expression-only) are analyzed separately. Panel summaries are median fractions and median context fraction. Two-sided equal-tail P=min(1,2*min((1+#null>=obs)/(N+1),(1+#null<=obs)/(N+1))) for component questions; context uses the requested upper-tail P. BH covers all 12 primary panel tests. Per-gene BH covers 17×5=85 two-sided tests. With 100 candidates, the smallest two-sided P is 2/101: individual-gene multiplicity power is limited; a non-significant result is not equivalence. Gene and panel nulls condition on this sampled-gene universe, not independent biological samples.

**FACT:**

{markdown_table(primary)}

Individual-gene patterns ranked by adjusted P then absolute candidate difference (descriptive if q>=0.05):

{markdown_table(patterns)}

## A5.5: compositional space
Square-root fractions define a Hellinger geometry and allow exact zeros without pseudocounts. PCA is fitted to eligible background; a seeded uniform sample of 5,000 genes is displayed along with matched candidates and all clock genes. Candidate-centroid connectors are descriptive. Formal secondary comparisons use clock-panel transformed centroids versus the matched-panel centroid cloud; null distances use leave-one-panel-out centroids. Raw-fraction Euclidean results are an explicit sensitivity, not a substitute for the primary geometry. BH covers four secondary tests.

{markdown_table(comp)}

## A5.6: relationship to frozen A2
All 17 genes are included; NFIL3, RORA, CRY1 and HLF are annotations only. Twelve Spearman associations and nominal BH are exported. Intervals use 2,000 paired-gene bootstrap samples and are descriptive because clock genes are correlated. Full donor/tissue-deletion correlation ranges are exported. A2 uses a different effect definition, with some scores on all 2,030 observations and fine cell identity; A5 uses supported major-lineage observations. Agreement is same-data methodological corroboration, not independent biological replication.

{markdown_table(a2sel)}

## A5.7: dependency and robustness
Delete each of the 18 supported donors and 29 supported tissues. Retain originally supported pairs during deletion (donor support may decrease to two), matching the frozen A2 deletion policy. Also reselect support >=2 and >=4 from the original 1,302 lineage observations; use donor-first main-effect allocation; compare all saved matching variants; remove targets above expression/variance background P95/P99 or prevalence below 0.5/0.8 with the corresponding slots removed from every null panel. These are sensitivities only; the primary panel never changes. An empty surviving panel is not tested. All effects and P values are cached and exported.

{markdown_table(rob)}

Robust panel findings under the recorded direction/significance rule:

{markdown_table(strong) if len(strong) else 'None met every prespecified robustness criterion.'}

## A5.8: negative control and conditional associations
Pairwise Spearman associations for expression, variance, pseudobulk prevalence, marginal effects and components are exported. Multivariable rank-linear regressions adjust logmean/logvariance ranks and prevalence; because logarithms preserve ranks, original-value ranks are identical. Variance incremental R2 and residual rank correlation ask whether association remains after mean/prevalence adjustment. There are no gene-independence P claims or causal interpretations. This cannot answer the stronger question conditioning on unavailable true single-cell detection.

{markdown_table(negative)}

Among clock-like candidate genes, `clock_like_dominance_summary.csv` describes which technical regimes accompany each largest variance component. This is association, not a mechanism. Candidate composition remains heterogeneous at comparable abundance/variance.

Within the 1,604 primary candidate genes, the same conditional rank model gives a much smaller, positive variance–lineage partial correlation (0.084) instead of the whole-universe negative coefficient. Thus the conditional relationship depends on the expression regime and cannot be reduced to a universal effect of variance. Technical predictors explain only 14.3% of rank variation in lineage fraction, 8.6% in interaction fraction, and 1.6% in context fraction in this restricted pool; most composition heterogeneity remains unexplained by these technical summaries.

{markdown_table(read('statistics/clock_like_multivariable.csv'))}

## A5.9: heterogeneity
All 17 clock genes are retained. Continuous fractions are primary; no post-hoc subgroup replaces a panel. The accompanying exploratory canonical-family table uses the original positive-limb/negative-limb definitions; auxiliary nuclear receptors and output genes are described separately without a new significance search. A largest-component label, when supplied, means only the arithmetic argmax and is not a discrete biological class.

{markdown_table(family_frame)}

## Figures and reproducibility
Eight main figures are exported as PNG (300 dpi), PDF and SVG, with source CSV for each. Numerical tests cover pure additive/interaction/residual signals, unbalanced designs, direct least-squares residual agreement, shift/scale/row-order invariance, zero variance and empirical-tail ties. `00_audit/final_validation.json` records tests, visual QA and unchanged hashes. No prior figure/report is overwritten.

## FACT / INTERPRETATION / NOT SUPPORTED
**FACT:** {int((primary.BH12<.05).sum())}/12 primary panel tests and {int((gene.BH85<.05).sum())}/85 primary gene-component tests meet BH<0.05. {len(strong)} panel tests meet the complete recorded robustness rule.

**INTERPRETATION:** The result concerns allocation of static expression variation after abundance/total-variance matching under the specified finite-design hierarchy. Components describe association with observed metadata, not uniquely separated biological drivers.

**NOT SUPPORTED:** temporal conclusions, causal allocation, independence of correlated clock genes, true single-cell detection adjustment for the broad universe, absence of architecture from non-significance, or a change to the frozen Phase A gate. Results depend on the chosen estimand and hierarchy; sensitivities quantify that dependence.

## Final seven questions
1. **Are clock genes more variable than ordinary expressed genes?** Yes descriptively. Core and Expanded median total variances are 0.02616 and 0.02222 versus 0.0000348 in eligible background; their median gene variance percentiles are 90.96 and 90.04. Median expression percentiles are 92.29 and 89.91. The permissive mean>0/variance>1e-12 background contains many extremely low-expression genes, so these are distributional context, not a specificity test.

2. **Is their high variance largely explained by their expression regime?** It is largely statistically associated with that regime: the background mean/prevalence model explains 97.87% of log-variance variation. Clock residual variance percentiles after adjustment range from 25.7 to 97.1; RORB remains at 97.1, whereas ARNTL and CLOCK are 45.6 and 29.4. This is not a causal explanation. Marginal variance–lineage association does not remain positive in the rank-linear mean/prevalence-adjusted model: partial rank correlation is -0.249 (incremental R2 0.0227), with strong predictor collinearity (condition number 26.9). This supports caution about interpreting total variance alone, not a claim that variance suppresses lineage biology. True single-cell detection adjustment cannot be performed broadly.

3. **What fraction is attributable to lineage, tissue, donor, lineage×tissue and residual structure?** Under the stated finite-design allocation, unweighted mean per-gene fractions are Core **24.58%, 12.06%, 5.81%, 9.56%, 47.99%**; Expanded **25.19%, 11.49%, 4.99%, 11.89%, 46.43%**, respectively. These mean fractions sum to 100%; the prespecified inference instead uses medians, whose components need not sum to one. These are allocated observed SS, not unique causal variance contributions.

4. **Given matched expression and total variance, is clock variance composition unusual?** Not demonstrated. No primary panel test passes BH (minimum q=0.623), and no gene-component test passes BH. Median context is Core 0.2245 versus null 0.2085 (upper-tail P=0.253) and Expanded 0.2175 versus 0.1997 (P=0.146). Hellinger centroid tests are also negative (Core P=0.757, Expanded P=0.488). All four matching variants, all 47 block deletions, support rules, alternative hierarchy and target-slot sensitivity runs retain no BH-significant panel findings. Modest interaction/context elevations recur directionally but are insufficient evidence of specificity; absence of significance is not equivalence.

5. **Is the clock system homogeneous?** No descriptively: NFIL3 and RORA allocate 53.3% and 48.2% to lineage; RORB allocates 43.3% to lineage×tissue and 52.7% to total context; HLF has 34.4% context, CRY1 21.1% tissue, and RORC 66.8% residual. These patterns do not meet matched individual-gene multiplicity criteria. Canonical-family summaries are exploratory and retain every gene; continuous scores are preferable to a forced discrete taxonomy.

6. **Does A5 independently support the A2 conserved-backbone / tissue-remodeling model?** It strongly corroborates the lineage-stability ordering by another calculation on the same data: rho=0.985, descriptive bootstrap interval [0.913,1.000], deletion range [0.961,0.993]. A2 tissue remodeling correlates with A5 tissue fraction (rho=0.706, interval [0.312,0.918], deletion range [0.593,0.725]). The broader context association is weaker (rho=0.449, interval crosses zero); A2 interaction versus A5 interaction is weak (rho=0.169). Thus there is partial same-data methodological support, strongest for the lineage backbone and tissue main effect, not independent replication of every layer.

7. **Is there evidence for clock-specific static transcriptional architecture beyond generic high-variance behavior?** This A5 analysis does not establish it after matched controls and correction. It supports heterogeneous, interpretable static expression architectures within clock genes, but those allocations are not unusually displaced from comparable nonclock genes under the recorded hierarchy. The classification below means specificity was not demonstrated under these tests, not that every possible clock-specific property is absent. A1/A2/A3, the current gate and Phase B remain untouched.

**{decision}**
'''
    (DEST/'reports/A5_VARIANCE_ARCHITECTURE_REPORT.md').write_text(text,encoding='utf-8')
    jwrite(DEST/'statistics/A5_decision.json',{'classification':decision,'primary_panel_significant':int((primary.BH12<.05).sum()),'gene_significant':int((gene.BH85<.05).sum()),'robust_panel_tests':strong[['panel','metric']].to_dict('records'),'gate_changed':False})
    env=__import__('clock_architecture_umap').original_spec();families=[]
    for g in clock.gene:
        family='positive_limb' if g in env['POSITIVE_LIMB'] else 'negative_limb' if g in env['NEGATIVE_LIMB'] else 'auxiliary_receptor' if g in ['NR1D1','NR1D2','RORA','RORB','RORC'] else 'output'
        families.append({'gene':g,'canonical_family':family,'largest_fraction':clock.set_index('gene').loc[g,FRAC].idxmax(),**clock.set_index('gene').loc[g,MET].to_dict()})
    write('statistics/exploratory_clock_families.csv',pd.DataFrame(families));write('statistics/exploratory_family_summary.csv',pd.DataFrame(families).groupby('canonical_family')[MET].median().reset_index())
    print(json.dumps(json.loads((DEST/'statistics/A5_decision.json').read_text()),indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):figures();report()
