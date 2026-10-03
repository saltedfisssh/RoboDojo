"""Offline video review with planner segments and exact source-frame selection."""

import json
from pathlib import Path
from urllib.parse import quote


def render_review_page(root, report):
    """Embed only accepted previews; all frame selectors refer to their original HDF5."""
    root = Path(root).resolve()
    tasks = []
    for task in report["tasks"]:
        episodes = []
        for episode in task["attempts"]:
            if "preview" not in episode or "duplicate_seed_of" in episode:
                continue
            if not all(episode[key] for key in ("complete", "success", "rgb", "terminal_state")):
                continue
            video = Path(episode["preview"])
            segments = video.with_suffix(".json")
            episodes.append(
                {
                    "seed": episode["seed"],
                    "frames": episode["frames"],
                    "fps": episode["frames"] / episode["duration_s"],
                    "source": episode["path"],
                    "video": quote(video.resolve().relative_to(root).as_posix(), safe="/"),
                    "within_budget": episode["within_benchmark_budget"],
                    "mode": episode["generation_mode"],
                    "segments": json.loads(segments.read_text()) if segments.exists() else [],
                }
            )
        tasks.append(
            {
                "name": task["task"],
                "instruction": task["instruction"],
                "count": task["successful_rgb_episodes"],
                "exported": task["exported"],
                "official": task["official_task_class_present"],
                "note": task.get("definition_note", ""),
                "episodes": episodes,
            }
        )
    payload = {
        "updated_at": report["updated_at"],
        "required": report["episodes_per_task"],
        "exported": report["exported_tasks"],
        "official_ready": report["ready_tasks"],
        "total": sum(task["count"] for task in tasks),
        "tasks": tasks,
        "launcher": str(Path(__file__).resolve().parents[2] / "scripts/datagen-uv.sh"),
    }
    # HTML script elements terminate at a literal closing tag even inside JSON.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    return _PAGE.replace("__REVIEW_DATA__", data)


_PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboDojo · 轨迹检查</title>
<style>
:root{font:15px/1.6 system-ui,sans-serif;color:#e5eaf1;background:#121820;color-scheme:dark}
*{box-sizing:border-box}body{margin:0}header{padding:22px 28px;border-bottom:1px solid #314151}
h1{font-size:25px;margin:0 0 8px}h2{font-size:21px;margin:0}p{margin:7px 0;color:#afbdce}
main{display:grid;grid-template-columns:285px minmax(0,1fr);min-height:80vh}
aside{padding:20px;border-right:1px solid #314151}article{padding:24px;min-width:0}
button,input,select{font:inherit;border:1px solid #42546a;border-radius:7px;background:#1e2a38;color:inherit}
button{cursor:pointer;padding:7px 12px}button:hover{background:#314359}button:disabled{opacity:.4;cursor:default}
button.active{border-color:#69d6c1;background:#203d3e;color:#a1f5e5}input,select{padding:8px;max-width:100%}
#search{width:100%;margin-bottom:9px}#filter{width:100%;margin-bottom:14px}
#tasks{display:flex;flex-direction:column;gap:6px}#tasks button{text-align:left;font-size:13px;overflow-wrap:anywhere}
.small{font-size:13px;color:#b6c4d2}.counts{display:flex;gap:20px;flex-wrap:wrap}
.counts strong{color:#a1f5e5;font-size:20px}.row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
#seeds{margin:18px 0 12px}video{width:100%;max-height:65vh;background:#070b10;border-radius:8px}
#controls{margin:12px 0}#frame{width:100px}#notice{color:#f1cc82}#segments{margin-top:15px;display:grid;gap:6px}
#segments button{text-align:left;display:grid;grid-template-columns:100px 1fr;gap:8px}
#segment-current{padding:12px 0;min-height:44px;color:#a1f5e5}#empty{padding:50px 0}
textarea{width:100%;min-height:80px;background:#111c28;color:#d3dfeb;border:1px solid #42546a;border-radius:7px}
a{color:#94c9ff}label{display:inline-flex;gap:6px;align-items:center}details{margin-top:16px}
@media(max-width:800px){main{display:block}aside{border-right:0;border-bottom:1px solid #314151}
#tasks{max-height:200px;overflow:auto}article{padding:16px}header{padding:18px}}
</style></head><body>
<header><h1>RoboDojo 轨迹检查</h1><div class="counts" id="counts"></div>
<p class="small" id="updated"></p>
<p class="small">只展示完整成功的三路相机轨迹。左侧为主相机，右上、右下分别为左腕、右腕相机。</p>
<a class="small" href="REVIEW.md">完整检查清单</a></header>
<main><aside><input id="search" type="search" placeholder="搜索任务" aria-label="搜索任务">
<select id="filter" aria-label="筛选任务"><option value="all">全部 35 类</option>
<option value="remaining">尚未达到目标数量</option><option value="ready">已有检查视频</option></select>
<nav id="tasks" aria-label="任务列表"></nav></aside>
<article><h2 id="title"></h2><p id="instruction"></p><p id="notice"></p>
<div class="row" id="seeds"></div><p id="empty" hidden>该任务还没有完整成功的检查视频。</p>
<div id="viewer"><video id="video" controls preload="metadata" playsinline></video>
<div class="row" id="controls"><button id="previous" title="上一帧">−1 帧</button>
<label>帧 <input id="frame" type="number" min="0" step="1" value="0"></label>
<button id="next" title="下一帧">+1 帧</button><span id="clock" class="small"></span>
<label>速度 <select id="speed"><option>0.5</option><option selected>1</option><option>2</option><option>4</option>
</select>×</label></div><p class="small" id="episode-info"></p>
<div id="segment-current" aria-live="polite"></div>
<details><summary>在 Isaac 中恢复当前帧</summary>
<p class="small">先暂停视频并选择帧，再复制命令到 RoboDojo 环境运行。命令仅恢复并核验状态。</p>
<textarea id="restore" readonly aria-label="当前帧恢复命令"></textarea>
<button id="copy">复制恢复命令</button><span id="copy-status" class="small"></span></details>
<details open><summary>原子任务时间轴 · 点击定位</summary><div id="segments"></div></details>
</div></article></main>
<script id="data" type="application/json">__REVIEW_DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id), video=$('video');
let task=null, episode=null, segmentButtons=[], selectedFrame=0, pendingSeek=null;
const shellQuote=s=>"'"+s.replaceAll("'", "'\\''")+"'";
const time=s=>`${Math.floor(s/60)}:${(s%60).toFixed(2).padStart(5,'0')}`;
const status=`${data.total} 条完整成功 RGB · ${data.exported}/35 类已导出 · `+
 `${data.official_ready}/35 类满足官方定义及数量`;
$('counts').textContent=status;
$('updated').textContent=`目标：每类 ${data.required} 个不同 seed。更新于 ${data.updated_at}；刷新页面获取新结果。`;
function listTasks(){
 const query=$('search').value.toLowerCase(), filter=$('filter').value; $('tasks').replaceChildren();
 for(const t of data.tasks){
  if(!t.name.toLowerCase().includes(query) || (filter==='remaining' && t.count>=data.required)
     || (filter==='ready' && !t.episodes.length))continue;
  const b=document.createElement('button');
  b.textContent=`${t.name}  ${t.count}/${data.required}`;
  b.classList.toggle('active',t===task); b.onclick=()=>chooseTask(t); $('tasks').append(b);
 }
}
function chooseTask(t){
 task=t; $('title').textContent=t.name; $('instruction').textContent=t.instruction;
 $('notice').textContent=[!t.official?'重建任务定义，未经官方验证。':'',t.note].filter(Boolean).join(' ');
 $('seeds').replaceChildren();
 for(const e of t.episodes){const b=document.createElement('button');b.textContent=`Seed ${e.seed}`;
  b.onclick=()=>chooseEpisode(e);b.dataset.seed=e.seed;$('seeds').append(b);}
 chooseEpisode(t.episodes[0]||null); listTasks();
}
function chooseEpisode(e){
 video.pause();episode=e;selectedFrame=0;pendingSeek=0;
 $('viewer').hidden=!e; $('empty').hidden=!!e;
 for(const b of $('seeds').children)b.classList.toggle('active',!!e && Number(b.dataset.seed)===e.seed);
 $('segments').replaceChildren();segmentButtons=[];
 if(!e){video.removeAttribute('src');video.load();return;}
 video.src=e.video; video.playbackRate=Number($('speed').value); $('frame').max=e.frames-1;
 const modes={physical_collection:'直接物理采集',
  checkpoint_render_of_successful_physical_plan:'成功物理轨迹逐帧补图',
  physical_replay_of_successful_plan:'成功动作序列物理重放'};
 const mode=modes[e.mode]||e.mode;
 const budget=e.within_budget===true?'在官方步数预算内':e.within_budget===false?'超出官方步数预算':'预算未记录';
 $('episode-info').textContent=`${e.frames} 帧 · ${time(e.frames/e.fps)} · ${mode} · ${budget}`;
 for(const s of e.segments){
  const b=document.createElement('button'), span=document.createElement('span'), text=document.createElement('span');
  span.textContent=time(s.start_frame/e.fps);text.textContent=s.text||s.primitive;
  b.append(span,text);b.onclick=()=>seek(s.start_frame);$('segments').append(b);segmentButtons.push([s,b]);
 }
 update(0);
}
function update(frame){
 if(!episode)return;selectedFrame=Math.min(episode.frames-1,Math.max(0,frame));
 if(document.activeElement!==$('frame'))$('frame').value=selectedFrame;
 $('clock').textContent=`/ ${episode.frames-1} · ${time(selectedFrame/episode.fps)}`;
 let active=null;
 for(const [s,b] of segmentButtons){const yes=selectedFrame>=s.start_frame && selectedFrame<s.end_frame;
  b.classList.toggle('active',yes); if(yes)active=s;}
 $('segment-current').textContent=active?`${active.primitive} · ${active.text}`:'';
 $('restore').value=`bash ${shellQuote(data.launcher)} replay --input ${shellQuote(episode.source)}`+
  ` --frame ${selectedFrame} --steps 0 --headless --no-images`;
 $('copy-status').textContent='';
}
function seek(frame){
 if(!episode||!Number.isFinite(frame))return;video.pause();
 const f=Math.min(episode.frames-1,Math.max(0,Math.round(frame)));pendingSeek=f;
 if(video.readyState>=1){video.currentTime=(f+0.01)/episode.fps;pendingSeek=null;}update(f);$('frame').value=f;
}
video.addEventListener('loadedmetadata',()=>{if(pendingSeek!==null)seek(pendingSeek);});
video.addEventListener('timeupdate',()=>{if(episode)update(Math.floor(video.currentTime*episode.fps+1e-5));});
$('frame').onchange=()=>seek(Number($('frame').value));
$('previous').onclick=()=>seek(selectedFrame-1);$('next').onclick=()=>seek(selectedFrame+1);
$('speed').onchange=()=>{video.playbackRate=Number($('speed').value);};
$('search').oninput=listTasks;$('filter').onchange=listTasks;
$('copy').onclick=async()=>{
 try{await navigator.clipboard.writeText($('restore').value);$('copy-status').textContent='已复制';}
 catch{ $('restore').focus();$('restore').select();$('copy-status').textContent='请复制已选中的命令';}
};
chooseTask(data.tasks.find(t=>t.episodes.length)||data.tasks[0]);
</script></body></html>"""
