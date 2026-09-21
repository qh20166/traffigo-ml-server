# -*- coding: utf-8 -*-
"""
admin_ui.py — Trang quản trị /admin của TraffiGo ML Server.

Đăng nhập bằng mật khẩu (biến môi trường TRAFFIGO_ADMIN_PASSWORD, mặc định
traffigo2026) -> dashboard gồm: tổng quan model/refresher/request, bảng đoạn
đường live, điều khiển refresher (quét ngay / tạm dừng / đổi chu kỳ).

Cách gắn: trong app.py, SAU KHI đã tạo `app` và các phần khác, gọi:
    from admin_ui import register_admin
    register_admin(app)
"""
import os
import secrets
import time
from typing import Dict

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

ADMIN_PASSWORD = os.environ.get("TRAFFIGO_ADMIN_PASSWORD", "traffigo2026")
_ADMIN_TOKENS: set = set()
_START_TIME = time.time()
_REQUEST_STATS: Dict[str, int] = {"total": 0}


def register_admin(app, state) -> None:
    """Gắn các endpoint quản trị + middleware đếm request vào app.

    state: dict chia sẻ với app.py chứa tham chiếu cần điều khiển:
      REFRESH_STATE, REFRESH_CFG, _refresh_wakeup, _tomtom_keys,
      MONITORED_SEGMENTS, LIVE_DATA, SPEED_MODEL, CLUSTER, MODEL_META
    """

    @app.middleware("http")
    async def _count_requests(request: Request, call_next):
        _REQUEST_STATS["total"] += 1
        key = request.url.path
        _REQUEST_STATS["count:" + key] = _REQUEST_STATS.get("count:" + key, 0) + 1
        return await call_next(request)

    def _is_admin(request: Request) -> bool:
        return request.cookies.get("admin_token") in _ADMIN_TOKENS

    @app.post("/admin/login")
    def admin_login(request: Request, payload: Dict = None):
        password = (payload or {}).get("password", "")
        if password == ADMIN_PASSWORD:
            token = secrets.token_hex(16)
            _ADMIN_TOKENS.add(token)
            resp = JSONResponse({"ok": True})
            resp.set_cookie("admin_token", token, max_age=86400, httponly=True)
            return resp
        return JSONResponse({"ok": False, "error": "Sai mật khẩu"}, status_code=401)

    @app.get("/admin/logout")
    def admin_logout(request: Request):
        _ADMIN_TOKENS.discard(request.cookies.get("admin_token"))
        resp = JSONResponse({"ok": True})
        resp.set_cookie("admin_token", "", max_age=0)
        return resp

    @app.get("/admin/api/overview")
    def admin_overview(request: Request):
        if not _is_admin(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        top = sorted(((k[6:], v) for k, v in _REQUEST_STATS.items() if k.startswith("count:")),
                     key=lambda x: -x[1])[:6]
        return {
            "uptimeSec": int(time.time() - _START_TIME),
            "requestsTotal": _REQUEST_STATS["total"],
            "requestsTop": [{"path": p, "count": v} for p, v in top],
            "model": {"loaded": state["SPEED_MODEL"] is not None,
                      "metrics": state["MODEL_META"].get("metrics")},
            "cluster": {"loaded": state["CLUSTER"] is not None},
            "refresher": {**state["REFRESH_STATE"],
                          "intervalSec": state["REFRESH_CFG"]["interval"],
                          "monitored": len(state["MONITORED_SEGMENTS"]),
                          "stored": len(state["LIVE_DATA"])},
            "tomtomKeys": len(state["_tomtom_keys"]),
        }

    @app.post("/admin/refresher/toggle")
    def admin_refresher_toggle(request: Request):
        if not _is_admin(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        state["REFRESH_CFG"]["enabled"] = not state["REFRESH_CFG"]["enabled"]
        state["REFRESH_STATE"]["enabled"] = state["REFRESH_CFG"]["enabled"]
        if state["REFRESH_CFG"]["enabled"] and state["_refresh_wakeup"]:
            state["_refresh_wakeup"].set()
        return {"enabled": state["REFRESH_CFG"]["enabled"]}

    @app.post("/admin/refresher/run-now")
    def admin_refresher_run_now(request: Request):
        if not _is_admin(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        state["REFRESH_CFG"]["enabled"] = True
        state["REFRESH_STATE"]["enabled"] = True
        if state["_refresh_wakeup"]:
            state["_refresh_wakeup"].set()
        return {"ok": True, "message": "Đã yêu cầu quét ngay"}

    @app.post("/admin/refresher/interval")
    def admin_refresher_interval(request: Request, payload: Dict = None):
        if not _is_admin(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            v = int((payload or {}).get("seconds", state["REFRESH_CFG"]["interval"]))
        except (TypeError, ValueError):
            v = state["REFRESH_CFG"]["interval"]
        state["REFRESH_CFG"]["interval"] = max(60, min(86400, v))
        if state["_refresh_wakeup"]:
            state["_refresh_wakeup"].set()
        return {"intervalSec": state["REFRESH_CFG"]["interval"]}

    @app.get("/admin/api/live")
    def admin_live(request: Request):
        if not _is_admin(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        segs = sorted(state["LIVE_DATA"].values(),
                      key=lambda s: s.get("fetchedAt", ""), reverse=True)
        return {"segments": segs[:300], "stored": len(state["LIVE_DATA"])}

    @app.get("/admin", response_class=HTMLResponse)
    def admin_page():
        return _DASH_HTML


_DASH_HTML = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TraffiGo - Quản trị</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:'Segoe UI',Arial,sans-serif}
body{background:#F3F1FB;min-height:100vh;padding:24px;color:#111}
h1{font-size:22px;color:#5C4FE0;margin-bottom:4px}
.sub{color:#888;font-size:13px;margin-bottom:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:18px}
.card{background:#fff;border-radius:16px;padding:16px;box-shadow:0 1px 4px rgba(60,40,150,.08)}
.card .lb{font-size:10px;font-weight:700;letter-spacing:.08em;color:#888}
.card .v{font-size:22px;font-weight:700;margin-top:4px}
.card .s{font-size:12px;color:#888;margin-top:2px}
.purple{color:#5C4FE0}.green{color:#2E7D32}.red{color:#C62828}
button{background:#5C4FE0;color:#fff;border:0;border-radius:24px;padding:9px 18px;font-weight:600;cursor:pointer;font-size:13px}
button.gray{background:#EEE;color:#444}
button:active{opacity:.85}
input{border:1px solid #E0DEF0;border-radius:10px;padding:9px 12px;font-size:14px;width:110px}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 1px 4px rgba(60,40,150,.08);margin-bottom:8px}
th{background:#5C4FE0;color:#fff;font-size:11px;letter-spacing:.06em;padding:10px 12px;text-align:left}
td{padding:9px 12px;border-bottom:1px solid #F0EEF8;font-size:13px}
.badge{padding:2px 10px;border-radius:10px;font-size:11px;font-weight:700}
.bj{background:#FFEBEE;color:#C62828}.bm{background:#FFF3E0;color:#E65100}.bc{background:#E8F5E9;color:#2E7D32}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:18px}
.section{font-size:15px;font-weight:700;margin:20px 0 10px;color:#333}
#login{max-width:360px;margin:80px auto;background:#fff;border-radius:18px;padding:28px;box-shadow:0 4px 20px rgba(60,40,150,.15);text-align:center}
#login input{width:100%;margin:14px 0}
.err{color:#C62828;font-size:13px;min-height:16px}
.top{display:flex;justify-content:space-between;align-items:center;gap:10px}
</style></head><body>
<div id="app"></div>
<script>
const $=id=>document.getElementById(id);
async function api(path,opt){const r=await fetch(path,{credentials:'same-origin',...opt});
 if(r.status===401){renderLogin();throw new Error('unauthorized')}return r.json()}
function fmtDur(s){const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?h+'h '+m+'p':m+'p'}
async function renderDash(){
 const o=await api('/admin/api/overview');const l=await api('/admin/api/live');
 const rf=o.refresher;
 const clBadge=s=>{const c=s.congestionIndex;const cls=c<0.7?'bj':(c<0.85?'bm':'bc');
  const lb=c<0.7?'Kẹt':(c<0.85?'Chậm':'Thoáng');return '<span class="badge '+cls+'">'+lb+'</span>'};
 $('app').innerHTML=`
 <div class="top"><div><h1>TraffiGo — Quản trị ML Server</h1>
 <div class="sub">Uptime ${fmtDur(o.uptimeSec)} · ${o.requestsTotal} request đã xử lý · ${o.tomtomKeys} khóa TomTom</div></div>
 <button class="gray" onclick="fetch('/admin/logout').then(()=>location.reload())">Đăng xuất</button></div>
 <div class="grid">
  <div class="card"><div class="lb">MODEL DỰ ĐOÁN</div><div class="v purple">${o.model.loaded?'Đã nạp':'Chưa nạp'}</div>
   <div class="s">MAE ${o.model.metrics?o.model.metrics.mae_kmh:'-'} km/h · R² ${o.model.metrics?o.model.metrics.r2:'-'}</div></div>
  <div class="card"><div class="lb">GOM CỤM K-MEANS</div><div class="v purple">${o.cluster.loaded?'k=6 ✓':'-'}</div>
   <div class="s">Dùng chung với app Android</div></div>
  <div class="card"><div class="lb">REFRESHER</div><div class="v ${rf.enabled?'green':'red'}">${rf.enabled?'Đang chạy':'Tạm dừng'}</div>
   <div class="s">Chu kỳ ${rf.intervalSec}s · kế tiếp ${rf.nextAt||'-'}</div></div>
  <div class="card"><div class="lb">ĐOẠN LIVE</div><div class="v">${rf.lastOk}/${rf.monitored}</div>
   <div class="s">OK/Fail chu kỳ #${rf.cycles} · lưu ${rf.stored}</div></div>
 </div>
 <div class="section">Điều khiển refresher</div>
 <div class="controls">
  <button onclick="act('/admin/refresher/run-now','Đã yêu cầu quét ngay')">⟳ Quét ngay</button>
  <button class="gray" onclick="act('/admin/refresher/toggle','Đã đổi trạng thái')">${rf.enabled?'⏸ Tạm dừng':'▶ Chạy lại'}</button>
  <input id="itv" type="number" value="${rf.intervalSec}" min="60"> <span style="font-size:13px">giây</span>
  <button class="gray" onclick="saveInterval()">Lưu chu kỳ</button>
 </div>
 <div class="section">Request nhiều nhất</div>
 <table><tr><th>Endpoint</th><th>Số lần</th></tr>
 ${(o.requestsTop||[]).map(r=>'<tr><td>'+r.path+'</td><td>'+r.count+'</td></tr>').join('')||'<tr><td colspan=2>Chưa có</td></tr>'}</table>
 <div class="section">Đoạn đường giám sát (${l.stored} đoạn có dữ liệu)</div>
 <table><tr><th>#</th><th>Đường</th><th>Live</th><th>FreeFlow</th><th>Trạng thái</th><th>Cập nhật</th></tr>
 ${(l.segments||[]).map((s,i)=>'<tr><td>'+(i+1)+'</td><td>'+s.name+'</td><td>'+Math.round(s.currentSpeed)+' km/h</td>'
  +'<td>'+Math.round(s.freeFlowSpeed)+'</td><td>'+clBadge(s)+'</td>'
  +'<td>'+((s.fetchedAt||'').slice(11))+'</td></tr>').join('')||'<tr><td colspan=6>Chưa có dữ liệu live</td></tr>'}</table>`;
}
async function act(p,msg){await api(p,{method:'POST'});renderDash()}
async function saveInterval(){await api('/admin/refresher/interval',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify({seconds:+$('itv').value})});renderDash()}
function renderLogin(){
 $('app').innerHTML='<div id="login"><h1>TraffiGo — Quản trị</h1>'
 +'<div class="sub">Đăng nhập nhà quản trị</div>'
 +'<input id="pw" type="password" placeholder="Mật khẩu admin" onkeydown="if(event.key===\\'Enter\\')login()">'
 +'<div class="err" id="err"></div>'
 +'<button style="width:100%" onclick="login()">Đăng nhập</button></div>'}
async function login(){
 const r=await fetch('/admin/login',{method:'POST',credentials:'same-origin',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({password:$('pw').value})});
 if(r.ok){renderDash()}else{$('err').textContent='Sai mật khẩu'}}
renderLogin();setInterval(()=>{try{renderDash()}catch(e){}},30000);
</script></body></html>"""
