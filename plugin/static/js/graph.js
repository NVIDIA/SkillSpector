function renderGraph(){
  const el=document.getElementById('cy');
  if(!el) return;
  // avoid re-init
  if(el.dataset.rendered==="1") return;
  const raw=document.getElementById('graph-data');
  if(!raw) return;
  let data;
  try{ data=JSON.parse(raw.textContent); }catch(e){ console.error(e); return; }
  if(!data.nodes || !data.nodes.length){
    el.innerHTML='<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#8b949e">No dependencies detected — isolated skills</div>';
    return;
  }
  const cy=cytoscape({
    container: el,
    elements: [...data.nodes, ...data.edges],
    style:[
      {selector:'node', style:{'background-color':'#76B900','label':'data(label)','color':'#e6edf3','font-size':'12px','text-valign':'center','text-halign':'center','width':'40px','height':'40px','border-width':'2px','border-color':'#1f2a37'}},
      {selector:'node[external]', style:{'background-color':'#ff9500'}},
      {selector:'edge', style:{'width':2,'line-color':'#1f2a37','target-arrow-color':'#1f2a37','target-arrow-shape':'triangle','curve-style':'bezier'}},
      {selector:'edge[type="external"]', style:{'line-color':'#ff9500','target-arrow-color':'#ff9500','line-style':'dashed'}}
    ],
    layout:{name:'cose', idealEdgeLength:100, nodeOverlap:20, refresh:20, fit:true, padding:30}
  });
  cy.on('tap','node', e=>{ const n=e.target; alert(n.id() + ' — in:'+ n.indegree() + ' out:'+ n.outdegree()); });
  el.dataset.rendered="1";
}
document.addEventListener('DOMContentLoaded', renderGraph);
window.renderGraph=renderGraph;
