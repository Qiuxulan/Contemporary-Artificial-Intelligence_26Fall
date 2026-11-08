"""从实验原始记录生成报告引用的七张400dpi PNG图片。"""
import os
import sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/project2-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from search.astar import solve_maze, solve_weighted, zero_heuristic  # noqa: E402
from search.alphabeta import build_sample_tree, minimax  # noqa: E402
from search.treasure_game import TreasureGame, GRID  # noqa: E402

OUT=ROOT/'figures'
COLORS=['#414B9E','#228C8D','#D5A85B','#852736','#9792CB','#64748B']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                     'axes.titleweight':'bold','axes.labelcolor':'#334155','text.color':'#25334A',
                     'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#CBD5E1',
                     'grid.color':'#DCE3ED','grid.alpha':.6,'figure.facecolor':'white',
                     'axes.prop_cycle':matplotlib.cycler(color=COLORS),'pdf.fonttype':42,'svg.fonttype':'none'})


# 默认不画整张图的总标题（报告里用中文图注）；需要时设置环境变量 FIG_TITLES=1
SHOW_SUPTITLE=os.environ.get('FIG_TITLES')=='1'


def save(fig,name,title):
    if SHOW_SUPTITLE:
        fig.suptitle(title,fontsize=16,fontweight='bold')
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/f'{name}.png',dpi=400,bbox_inches='tight',pad_inches=0,facecolor='white')
    plt.close(fig)


def values(rows,metric):
    return np.array([float(r[metric]) for r in rows if r.get(metric,'')!=''])


def select(rows,**conditions):
    return [r for r in rows if all(str(r.get(k,''))==str(v) for k,v in conditions.items())]


def ordered(rows,key):
    return list(dict.fromkeys(r[key] for r in rows))


def grid_plot(ax,grid,path=None,expanded=None,title=''):
    n,m=len(grid),len(grid[0]);data=np.ones((n,m))
    for r in range(n):
        for c in range(m):
            cell=str(grid[r][c]);data[r,c]=np.nan if cell=='X' else (float(cell) if cell.isdigit() else 1)
    cmap=plt.get_cmap('YlOrBr').copy();cmap.set_bad('#28364B')
    ax.imshow(data,cmap=cmap,vmin=0,vmax=9)
    if expanded:
        ys,xs=zip(*expanded);ax.scatter(xs,ys,s=260,c=COLORS[0],alpha=.22,marker='s')
    if path:
        ys,xs=zip(*path);ax.plot(xs,ys,'o-',color=COLORS[0],lw=2.5,ms=5,markeredgecolor='white')
    for r in range(n):
        for c in range(m):
            cell=str(grid[r][c])
            if cell not in ('.','X'):
                ax.text(c,r,cell,ha='center',va='center',fontsize=9,fontweight='bold',
                        bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':1})
    ax.set_xticks(np.arange(-.5,m,1),minor=True);ax.set_yticks(np.arange(-.5,n,1),minor=True)
    ax.grid(which='minor',color='white',lw=.8);ax.tick_params(which='minor',length=0)
    ax.set_xticks(range(m));ax.set_yticks(range(n));ax.set_title(title,pad=12)
    ax.set_xlabel('Column');ax.set_ylabel('Row')


def draw_maps():
    maze=[['S','.','.','X','.'],['.','X','.','.','.'],['.','X','.','X','.'],
          ['.','.','.','X','E'],['X','X','.','.','.']]
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),layout='constrained')
    for ax,label,kw in zip(axes,['Dijkstra','A* / FIFO','A* / deeper first'],
                            [{'heuristic':zero_heuristic},{},{'tiebreak':'deep'}]):
        path,s=solve_maze((0,0),(3,4),maze,**kw)
        grid_plot(ax,maze,path,s['expanded_set'],f'{label}\n{s["expanded"]} expanded | {s["cost"]:.0f} steps')
    save(fig,'fig1_maze','01  |  Search footprint and shortest path')
    grid=[['S',1,1,8,1],[1,8,3,1,1],[1,8,1,8,1],[1,1,1,8,1],[8,8,1,1,'E']]
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    cost,path,s=solve_weighted((0,0),(4,4),grid)
    grid_plot(axes[0],grid,path,s['expanded_set'],f'A* with admissible h | total cost {cost:g}')
    costs=[0]+[0 if grid[r][c]=='E' else grid[r][c] for r,c in path[1:]]
    axes[1].bar(range(len(path)),costs,color=COLORS[2],label='Entry cost')
    axes[1].plot(range(len(path)),np.cumsum(costs),'o-',color=COLORS[0],label='Cumulative cost')
    axes[1].set_xticks(range(len(path)),[f'{r},{c}' for r,c in path],rotation=45)
    axes[1].set_xlabel('Path coordinate (row, column)');axes[1].set_ylabel('Cost')
    axes[1].set_title('Cost Accumulation Along the Optimal Path',fontsize=11)
    axes[1].legend();axes[1].grid(axis='y')
    save(fig,'fig2_terrain_path_cost','02  |  Terrain weights and path-cost accounting')


def draw_tiebreak(rows):
    scenes=ordered(rows,'scenario');fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    x=np.arange(len(scenes))
    for j,method in enumerate(('fifo','deep')):
        data=[values(select(rows,scenario=s,method=method),'expanded') for s in scenes]
        axes[0].bar(x+(j-.5)*.36,[np.mean(a) for a in data],.36,yerr=[np.std(a,ddof=1) if len(a)>1 else 0 for a in data],capsize=3,label=method,color=COLORS[j])
    axes[0].set_xticks(x,[s.replace(' p=','\np=') for s in scenes]);axes[0].set_ylabel('Expanded nodes (mean ± SD)')
    axes[0].legend();axes[0].grid(axis='y');axes[0].set_title('Same map, same heuristic, different ties')
    reductions=[]
    for scene in scenes:
        f={r['seed']:float(r['expanded']) for r in select(rows,scenario=scene,method='fifo')}
        d={r['seed']:float(r['expanded']) for r in select(rows,scenario=scene,method='deep')}
        reductions.append([100*(1-d[k]/f[k]) for k in f])
    axes[1].boxplot(reductions,tick_labels=[s.replace(' p=','\np=') for s in scenes],showfliers=False)
    axes[1].axhline(0,lw=1,color=COLORS[3]);axes[1].set_ylabel('Paired reduction in expanded nodes (%)')
    axes[1].set_title('Variation across individual mazes');axes[1].grid(axis='y')
    save(fig,'fig4_tiebreak','04  |  Tie-breaking with identical optimal path costs')


def draw_tree():
    root=build_sample_tree();coords={};nodes={};edges=[];leaves=[0]
    def layout(node,path=(),depth=0):
        nodes[path]=node
        if node.is_leaf:x=leaves[0];leaves[0]+=1
        else:
            xs=[]
            for i,c in enumerate(node.children):
                child=path+(i,);xs.append(layout(c,child,depth+1));edges.append((path,child))
            x=float(np.mean(xs))
        coords[path]=(x,-depth);return x
    layout(root);seen=set();backups={};bounds=set()
    def trace(node,path,is_max,alpha,beta):
        seen.add(path)
        if node.is_leaf:return node.value
        value=-np.inf if is_max else np.inf
        for i,c in enumerate(node.children):
            v=trace(c,path+(i,),not is_max,alpha,beta)
            value=max(value,v) if is_max else min(value,v)
            if is_max:alpha=max(alpha,value)
            else:beta=min(beta,value)
            if alpha>=beta:
                bounds.add(path);break
        backups[path]=value
        return value
    assert trace(root,(),True,-np.inf,np.inf)==5
    fig,ax=plt.subplots(figsize=(11,4.6),layout='constrained')
    for p,c in edges:
        ax.plot([coords[p][0],coords[c][0]],[coords[p][1],coords[c][1]],color='#CBD5E1' if c not in seen else '#64748B',ls='--' if c not in seen else '-',zorder=1)
    for path,node in nodes.items():
        x,y=coords[path];v=minimax(node,len(path)%2==0,{'visited':0})
        color=COLORS[len(path)%2] if path in seen else '#E2E8F0'
        ax.scatter(x,y,s=700,color=color,zorder=2,edgecolor='white',linewidth=2)
        label=('≥' if len(path)%2==0 else '≤')+str(backups[path]) if path in bounds else str(v)
        ax.text(x,y,label,ha='center',va='center',color='white' if path in seen else '#64748B',fontweight='bold',zorder=3)
        if path not in seen:ax.text(x,y-.27,'pruned',ha='center',fontsize=9,color=COLORS[3])
    ax.set_yticks([0,-1,-2,-3],['MAX','MIN','MAX','LEAF']);ax.set_xticks([])
    ax.set_ylim(-3.5,.45);ax.spines[['left','bottom']].set_visible(False)
    ax.set_title('Root value = 5  |  Minimax: 15 visits  |  Alpha-Beta: 14 visits',fontsize=11)
    save(fig,'fig7_tree','07  |  Value backup and the pruned leaf (left-to-right order)')


def render_all(rows):
    draw_maps()
    draw_tree()
    tiebreak = select(rows, experiment='tiebreak')
    if tiebreak:
        draw_tiebreak(tiebreak)
    render_report_figures(rows)


def render_report_figures(rows):
    """报告版使用较少面板、短标签和较大字号，保证缩入A4后仍可阅读。"""
    with plt.rc_context({'font.size':11,'axes.titlesize':11,'xtick.labelsize':10,'ytick.labelsize':10}):
        part=select(rows,experiment='ordering')
        if part:
            fig,axes=plt.subplots(1,2,figsize=(8.5,2.7),layout='constrained')
            scenes=ordered(part,'scenario');x=np.arange(len(scenes))
            for method,label,color in zip(ordered(part,'method'),['MM','AB natural','AB best*','AB worst*'],COLORS):
                effort=[np.mean(values(select(part,scenario=s,method=method),'visited')) for s in scenes]
                total=[np.mean(values(select(part,scenario=s,method=method),'time_ms'))+np.mean(values(select(part,scenario=s,method=method),'prep_ms')) for s in scenes]
                axes[0].plot(x,effort,'o-',label=label,color=color,ms=4)
                axes[1].plot(x,total,'o-',color=color,ms=4)
            for ax in axes:ax.set_xticks(x,['2/6','3/4','3/6','4/4','4/5']);ax.set_xlabel('Branching / depth');ax.set_yscale('log');ax.grid()
            axes[0].set_ylabel('Mean node visits');axes[1].set_ylabel('Total time (ms)')
            axes[0].legend(fontsize=8,ncol=2)
            axes[0].set_title('Node Visits by Search Strategy')
            axes[1].set_title('Total Time Including Sorting')
            save(fig,'report2_ordering_titled','Move ordering: search savings versus total cost')
        game=TreasureGame();names=list(game.dist)
        fig,axes=plt.subplots(1,3,figsize=(9.4,2.8),layout='constrained')
        grid_plot(axes[0],GRID,title='Entry costs')
        axes[0].set_xlabel('');axes[0].set_ylabel('');axes[0].set_xticks([]);axes[0].set_yticks([])
        for t in axes[0].texts:t.set_fontsize(10)
        matrix=np.array([[game.dist[a][b] for b in names] for a in names])
        axes[1].imshow(matrix,cmap='Blues');axes[1].set_xticks(range(6),names);axes[1].set_yticks(range(6),names)
        axes[1].set_title('Travel cost')
        for i in range(6):
            for j in range(6):axes[1].text(j,i,f'{matrix[i,j]:g}',ha='center',va='center',fontsize=10,color='white' if matrix[i,j]>7 else '#25334A')
        root=(game.key_points['A'],game.key_points['B'],tuple(sorted(game.values)),0,0)
        targets=[];data=[]
        for target,child in game.successors(root,True):
            targets.append(target);data.append([game.minimax(child,d-1,False,{'visited':0}) for d in range(1,5)])
        data=np.array(data);lim=np.max(np.abs(data));axes[2].imshow(data,cmap='RdBu',vmin=-lim,vmax=lim)
        axes[2].set_xticks(range(4),['1','2','3','4']);axes[2].set_yticks(range(4),targets)
        axes[2].set_xlabel('');axes[2].set_title('Exact utility A − B')
        for i in range(4):
            for j in range(4):axes[2].text(j,i,f'{data[i,j]:+g}',ha='center',va='center',fontsize=11,color='white' if abs(data[i,j])>lim*.7 else '#25334A',fontweight='bold' if data[i,j]==data[:,j].max() else 'normal')
        save(fig,'report3_game_no_depth_label','From terrain to competitive choices')
        part=select(rows,experiment='scale')
        if part:
            fig,axes=plt.subplots(1,2,figsize=(8.5,2.8),layout='constrained')
            for ax,scene in zip(axes,ordered(part,'scenario')):
                subset=select(part,scenario=scene);depths=sorted({int(r['depth']) for r in subset})
                for method,label in zip(ordered(subset,'method'),['MM','AB name','AB gain','Memoized']):
                    arrays=[values(select(subset,method=method,depth=d),'evaluated') for d in depths]
                    ax.errorbar(depths,[np.mean(a) for a in arrays],yerr=[np.std(a,ddof=1) if len(a)>1 else 0 for a in arrays],fmt='o-',ms=4,capsize=2,label=label)
                ax.set_title(scene.split(' / ')[1]);ax.set_xticks(depths);ax.set_yscale('log');ax.grid();ax.set_xlabel('Depth H')
            axes[0].set_ylabel('Evaluated nodes');axes[1].legend(fontsize=8,ncol=2)
            save(fig,'report4_scaling','Scaling: mean ± SD across eight maps')
