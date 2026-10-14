/* Polar Clock — UN-style azimuthal equidistant, N-pole center, Antarctica border.
   Sun = hour hand. Moon = second tide hand. No dependencies. */
const cv = document.getElementById('polar');
const ctx = cv.getContext('2d');
const state = { mode:'clock', day:24, rot:0, zoom:1, showNight:true, showMyth:true, showMarkets:true, showLat33:true, cal:'gregory' };

// ---------- astro helpers ----------
function dayOfYear(d){ return Math.floor((d - new Date(d.getFullYear(),0,0))/864e5); }
function sunDecl(d){ return -23.44*Math.cos(2*Math.PI*(dayOfYear(d)+10)/365); } // deg
function utcHours(d){ return d.getUTCHours()+d.getUTCMinutes()/60+d.getUTCSeconds()/3600; }
function sunLon(d){ // subsolar longitude, deg -180..180
  let lon = 180 - 15*utcHours(d);
  lon=((lon+540)%360)-180; return lon;
}
const SYNODIC=29.530588853, REF_NEW=Date.UTC(2000,0,6,18,14)/864e5;
function moonAge(d){ const j=d.getTime()/864e5; return ((j-REF_NEW)%SYNODIC+SYNODIC)%SYNODIC; }
function moonLon(d){ const age=moonAge(d); const elong=360*age/SYNODIC; let l=sunLon(d)+elong; return ((l+540)%360)-180; }
function moonLat(){ return 5.14*Math.sin(2*Math.PI*moonAge(new Date())/27.32); }
function moonIllum(d){ return (1-Math.cos(2*Math.PI*moonAge(d)/SYNODIC))/2; }

// projection: lat/lon -> canvas x,y (azimuthal equidistant, N pole center)
let CX=450, CY=450, RMAX=400;
function proj(lat,lon){
  const r=(90-lat)/180*RMAX*state.zoom;
  const a=(lon+state.rot)*Math.PI/180; // 0° at top, +east clockwise
  return [CX+r*Math.sin(a), CY-r*Math.cos(a)];
}
function circleLat(lat){ ctx.beginPath(); for(let lo=-180;lo<=180;lo+=2){ const [x,y]=proj(lat,lo); lo===-180?ctx.moveTo(x,y):ctx.lineTo(x,y);} ctx.closePath(); }

// ---------- drawing ----------
const MARKETS=[
 {n:'Sydney', lon:151.2, lat:-33.9, open:21, close:6},
 {n:'Tokyo', lon:139.7, lat:35.7, open:0, close:9},
 {n:'London', lon:-0.12, lat:51.5, open:7, close:16},
 {n:'New York', lon:-74, lat:40.7, open:12, close:21},
];
function sessOpen(m,h){ return m.open<m.close ? (h>=m.open&&h<m.close) : (h>=m.open||h<m.close); }
const ZODIAC13=[['Pisces',0],['Aries',30],['Taurus',60],['Gemini',90],['Cancer',115],['Leo',140],['Virgo',170],['Libra',205],['Scorpius',230],['Ophiuchus',250],['Sagittarius',270],['Capricornus',300],['Aquarius',330]];
const MYTH=[['Atlantis',-30,25,'mid-Atlantic'],['Hyperborea',0,75,'beyond north wind'],['Mt Meru / Zion',90,35,'axis + four rivers'],['Thule',-20,72,'ultima'],['Antipodes',180,-45,'under-feet land']];

function draw(now){
  const W=cv.width,H=cv.height; CX=W/2;CY=H/2;RMAX=Math.min(W,H)/2-28;
  ctx.clearRect(0,0,W,H);
  // space
  ctx.fillStyle='#04060d';ctx.fillRect(0,0,W,H);
  // stars in planetarium mode
  if(state.mode==='sky'){ ctx.fillStyle='#cfe0ff'; for(let i=0;i<220;i++){ const a=(i*137.5+state.rot)%360, r=(0.15+((i*89)%80)/100)*RMAX; const x=CX+r*Math.sin(a*Math.PI/180),y=CY-r*Math.cos(a*Math.PI/180); ctx.fillRect(x,y,1.4,1.4);} }

  // ocean disc
  const g=ctx.createRadialGradient(CX,CY,10,CX,CY,RMAX);
  g.addColorStop(0,'#dfe9ff');g.addColorStop(.12,'#b9cdf3');g.addColorStop(.5,'#33507e');g.addColorStop(.85,'#12233f');g.addColorStop(1,'#e8f2ff');
  ctx.fillStyle=g;ctx.beginPath();ctx.arc(CX,CY,RMAX,0,7);ctx.fill();

  // day/night terminator: shade where |lon - sunLon|>90
  if(state.showNight && state.mode==='clock'){
    ctx.save();ctx.beginPath();ctx.arc(CX,CY,RMAX,0,7);ctx.clip();
    ctx.fillStyle='rgba(2,6,20,0.55)';
    for(let la=-90;la<90;la+=3)for(let lo=-180;lo<180;lo+=3){
      let d=((lo-sunLon(now)+540)%360)-180; if(Math.abs(d)>90){ const [x,y]=proj(la,lo); ctx.fillRect(x-3,y-3,6,6);} }
    ctx.restore();
  }

  // latitude rings
  const rings=[[90,'N POLE · POLARIS ⏄'],[66.5,'Arctic'],[33.3,'LAT 33° home'],[23.4,'Tropic ♋'],[0,'Equator'],[-23.4,'Tropic ♑'],[-60,'60°S'],[-90,'ANTARCTICA']];
  ctx.font='11px system-ui';
  for(const [la,label] of rings){
    if(la===33.3&&!state.showLat33) continue;
    ctx.strokeStyle= la===33.3?'#ffd166': la===0?'#7fb2ff':'rgba(230,240,255,.5)';
    ctx.lineWidth= la===33.3?2.2:1; circleLat(la); ctx.stroke();
    const [lx,ly]=proj(la,state.rot*-1); ctx.fillStyle= la===33.3?'#ffd166':'#cfe0ff';
    ctx.fillText(la===90?label:`${la}° ${label}`, lx+6, ly-4);
  }
  // longitude spokes: 24h (15°) minor, 17h mode adds 17 spokes
  const nSp = state.day===17?17:24;
  for(let i=0;i<nSp;i++){
    const lon=-180+360*i/nSp;
    const [x0,y0]=proj(88,lon),[x1,y1]=proj(-88,lon);
    ctx.strokeStyle= state.day===17?'rgba(94,242,184,.55)':'rgba(200,220,255,.30)';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.stroke();
    const [tx,ty]=proj(-78,lon); ctx.fillStyle='#9fb4dd'; ctx.fillText(state.day===17?`H${i+1}`:`${((lon+360)%360/15).toFixed(0)}h`,tx-8,ty);
  }

  // Antarctica seconds/minutes border (watch-face)
  const nowS=now.getUTCSeconds()+now.getUTCMilliseconds()/1000, nowM=now.getUTCMinutes()+nowS/60;
  for(let s=0;s<60;s++){
    const a=s/60*2*Math.PI, big=s%5===0;
    const r1=RMAX-2, r0=RMAX-(big?22:12);
    ctx.strokeStyle= Math.floor(nowS)===s?'#ff5d5d':'#eaf3ff'; ctx.lineWidth=big?3:1.2;
    ctx.beginPath();
    ctx.moveTo(CX+r1*Math.sin(a),CY-r1*Math.cos(a));ctx.lineTo(CX+r0*Math.sin(a),CY-r0*Math.cos(a));ctx.stroke();
    if(big){ const m=s/5; const am=m/12*2*Math.PI;
      const [nx,ny]=[CX+(RMAX-34)*Math.sin(am),CY-(RMAX-34)*Math.cos(am)];
      ctx.fillStyle= Math.floor(nowM)%12===m?'#ffcf4d':'#0a1a33'; ctx.beginPath();ctx.arc(nx,ny,Math.floor(nowM)%12===m?11:9,0,7);ctx.fill();
      ctx.fillStyle= Math.floor(nowM)%12===m?'#111':'#eaf3ff';ctx.fillText(String(m===0?12:m),nx-4,ny+4);
    }
  }
  ctx.fillStyle='#9fb4dd';ctx.fillText(state.mode==='sky'?'13-sign zodiac border · Ophiuchus restored':'ANTARCTICA ice border = seconds (outer) + minutes (inner hours) · rotate = sidereal nudge',CX-250,CY+RMAX+16);

  // mythic overlay
  if(state.showMyth){
    ctx.font='12px system-ui';
    for(const [n,lo,la,sub] of MYTH){ const [x,y]=proj(la,lo);
      ctx.strokeStyle='#c9a7ff';ctx.setLineDash([5,4]);ctx.beginPath();ctx.arc(x,y,20,0,7);ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle='#e6d6ff';ctx.fillText(`◈ ${n}`,x+24,y);ctx.fillStyle='#9d8ac7';ctx.fillText(sub,x+24,y+13);
    }
    // four rivers from Meru
    const [mx,my]=proj(35,90); ctx.strokeStyle='#7fe7ff'; for(let k=0;k<4;k++){ ctx.beginPath();ctx.moveTo(mx,my); const a=k*Math.PI/2+.4; ctx.quadraticCurveTo(mx+50*Math.cos(a),my+50*Math.sin(a),mx+90*Math.cos(a),my+90*Math.sin(a)); ctx.stroke(); }
  }

  // markets
  if(state.showMarkets){
    const h=utcHours(now);
    for(const m of MARKETS){ const [x,y]=proj(m.lat,m.lon); const open=sessOpen(m,h);
      ctx.fillStyle=open?'#5ef2b8':'#3a4a6e'; ctx.beginPath();ctx.arc(x,y,open?9:6,0,7);ctx.fill();
      ctx.fillStyle='#dfe8ff';ctx.font='12px system-ui';ctx.fillText(`${open?'●':'○'} ${m.n}`,x+12,y+4);
    }
  }

  // planetarium zodiac ring
  const sl=sunLon(now), ml=moonLon(now);
  if(state.mode==='sky'){
    ctx.font='13px system-ui';
    ZODIAC13.forEach(([n,deg],i)=>{ const a=(deg-180+state.rot)*Math.PI/180; const x=CX+(RMAX+2)*Math.sin(a),y=CY-(RMAX+2)*Math.cos(a);
      ctx.fillStyle= i===9?'#ffb3ff':'#ffe9a8'; ctx.fillText(`${n} ♈♉♊…`[0]===' '?'':'✦ '+n,x-30,y); });
    // lunar months: 13 x 28d
    ctx.fillStyle='#8ea0c8';ctx.fillText('13 lunar months · Turtle-back year · each full moon = 1 month',CX-160,CY-RMAX-8);
  }

  // SUN hand (hour) — from center through subsolar point, extended to border
  const sd=sunDecl(now);
  const [sx,sy]=proj(sd,sl);
  ctx.strokeStyle='#ffcf4d';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(CX,CY);ctx.lineTo(sx,sy);ctx.stroke();
  const sa=Math.atan2(sx-CX,-(sy-CY)); const bx=CX+RMAX*Math.sin(sa),by=CY-RMAX*Math.cos(sa);
  ctx.strokeStyle='rgba(255,207,77,.5)';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(bx,by);ctx.stroke();
  ctx.fillStyle='#ffcf4d';ctx.beginPath();ctx.arc(sx,sy,10,0,7);ctx.fill();ctx.fillStyle='#221a00';ctx.font='bold 12px system-ui';ctx.fillText('☀',sx-6,sy+4);
  // MOON hand (currency tide)
  const [qx,qy]=proj(moonLat(),ml);
  ctx.strokeStyle='#bfe3ff';ctx.lineWidth=2.5;ctx.beginPath();ctx.moveTo(CX,CY);ctx.lineTo(qx,qy);ctx.stroke();
  ctx.fillStyle='#dff1ff';ctx.beginPath();ctx.arc(qx,qy,7,0,7);ctx.fill();ctx.fillStyle='#0a1a33';ctx.fillText('☾',qx-5,qy+4);

  // center: North Pole + Polaris
  ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(CX,CY,7,0,7);ctx.fill();
  ctx.fillStyle='#0a1a33';ctx.font='bold 9px system-ui';ctx.fillText('N',CX-4,CY+3);
  ctx.fillStyle='#ffe9a8';ctx.fillText('✶ Polaris above (fixed)',CX+12,CY-8);

  // second hand along border
  const sa2=nowS/60*2*Math.PI; ctx.strokeStyle='#ff5d5d';ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(CX,CY);ctx.lineTo(CX+(RMAX-26)*Math.sin(sa2),CY-(RMAX-26)*Math.cos(sa2));ctx.stroke();
}

// ---------- readouts ----------
function julianDate(d){ const g=new Date(d); g.setDate(g.getDate()-13); return g; } // 1900–2099 approx
function fmt17(d){ const mins=d.getUTCHours()*60+d.getUTCMinutes()+d.getUTCSeconds()/60; const h17=mins/(24*60)*17; const H=Math.floor(h17), M=Math.floor((h17-H)*60), S=Math.floor((((h17-H)*60)-M)*60); return `${String(H).padStart(2,'0')}:${String(M).padStart(2,'0')}:${String(S).padStart(2,'0')} /17h`; }
function calStr(d,cal){
  try{
    if(cal==='julian-cal'){ const j=julianDate(d); return `Julian ${j.toISOString().slice(0,10)} (Greg −13d)`; }
    return new Intl.DateTimeFormat('en-u-ca-'+cal,{dateStyle:'full'}).format(d);
  }catch{ return d.toUTCString(); }
}
const HOLIDAYS=[
 ['01-01',"New Year · Secular/bank (worldwide)"],
 ['01-06',"Epiphany · Christianity · + Armenian Christmas"],
 ['01-07',"Orthodox Christmas (Julian Dec 25)"],
 ['02-02',"Groundhog Day · Secular + Candlemas · Christianity + Imbolc · Pagan/Wicca"],
 ['02-14',"Valentine's · Secular/Christian legend"],
 ['03-08',"Holi (movable, spring) · Hinduism · + Women's Day"],
 ['03-17',"St Patrick's · Christianity/Ireland bank"],
 ['04-13',"Vaisakhi · Sikhism · + Songkran · Buddhism/Thailand"],
 ['04-20',"Ridván (Apr 20–May 2) · Baháʼí"],
 ['04-23',"St George's + Shakespeare's birthday (1564)"],
 ['05-23',"Declaration of the Báb · Baháʼí"],
 ['06-16',"Bloomsday (Ulysses, 1904) · literature"],
 ['07-04',"US Independence bank holiday"],
 ['08-15',"Assumption · Christianity + Obon · Buddhism/Japan"],
 ['09-22',"Mabon / equinox · Pagan + Hobbit Day (LOTR)"],
 ['10-02',"Gandhi Jayanti · India bank"],
 ['10-31',"Halloween · secular + Samhain · Pagan/Wicca"],
 ['11-01',"All Saints · Christianity + Día de Muertos (Nov 1–2)"],
 ['12-08',"Bodhi Day · Buddhism"],
 ['12-10',"Hanukkah (movable Kislev) · Judaism"],
 ['12-25',"Christmas · Christianity/bank + Newtonmas (1642, Julian)"],
 ['12-26',"Kwanzaa begins · African heritage + Boxing Day bank"],
 ['EID',"Eid al-Fitr / Eid al-Adha (movable, Shawwal/Dhu Hijjah) · Islam"],
 ['DIWALI',"Diwali (movable, Kartika) · Hinduism/Jainism/Sikhism"],
 ['PESACH',"Passover (movable, Nisan 15) · Judaism"],
 ['EASTER',"Easter (movable, first Sun after full moon) · Christianity"],
 ['VESAK',"Vesak (movable, Vaiśākha full moon) · Buddhism"],
 ['1938-04-18',"Action Comics #1 — Superman debut"],
 ['1940-05-27',"Batman #1 (Spring 1940) · comics"],
 ['1989-12-17',"The Simpsons first episode"],
 ['1999-01-31',"Family Guy first episode"],
 ['-2560',"Great Pyramid era · prehistory Sharpe"],
 ['-1200',"Torah compilation tradition · Judaism"],
 ['-0500',"Tripitaka oral tradition · Buddhism"],
 ['0030',"Sermon on the Mount tradition · Christianity"],
 ['0610',"First revelation · Islam (Qur'an)"],
 ['1611',"KJV Bible published"],
];
const QUOTES=[['Albert Einstein','Imagination is more important than knowledge.'],['Mahatma Gandhi','Be the change you wish to see in the world.'],['Peter Griffin','Freakin’ sweet. (Quote of the day: try the veal, respect the tides.)'],['Rumi','The wound is the place where the Light enters you.'],['Hypatia','Reserve your right to think.']];
const WORDS=[['es','¿Dónde está el baño?','Where is the bathroom?'],['ja','おはよう (ohayō)','Good morning'],['ar','سلام (salām)','Peace / hello'],['he','תודה (todah)','Thank you'],['hi','धन्यवाद (dhanyavaad)','Thank you'],['yo','Ẹ ku abọ̀','Welcome'],['mi','Kia ora','Hello / be well']];
const KNOTS=['Bowline — rescue loop that never jams','Clove hitch — fast lashing for posts','Figure-eight — stopper + follow-through tie-in','Sheet bend — join two ropes','Trucker’s hitch — 3:1 pulley for loads','Prusik — slide-and-grip friction hitch','Anchor bend — secure to a ring'];

function tideModel(d,lat=33.3){
  // semidiurnal: moon dominates, sun modulates. phase from moon/sun hour angles at observer lon 0 for demo + lat factor
  const h=utcHours(d), age=moonAge(d);
  const mPhase=2*Math.PI*((h - (moonLon(d)/15))/12.42), sPhase=2*Math.PI*((h-(sunLon(d)/15))/12.0);
  const latF=Math.cos(lat*Math.PI/180)+0.3;
  const ht= 1.1*Math.cos(mPhase)*latF + 0.45*Math.cos(sPhase)*latF;
  return ht; // meters-ish
}
function drawSeries(canvas,fn,color,unit){
  const c=canvas.getContext('2d');c.clearRect(0,0,canvas.width,canvas.height);
  c.strokeStyle='#22345e';c.beginPath();c.moveTo(0,70);c.lineTo(520,70);c.stroke();
  c.strokeStyle=color;c.lineWidth=2;c.beginPath();
  for(let i=0;i<=120;i++){ const d=new Date(Date.now()+(i-24)*3600e3); const v=fn(d); const x=i/120*520, y=70-v*32; i?c.lineTo(x,y):c.moveTo(x,y);} c.stroke();
  c.fillStyle='#8ea0c8';c.font='11px system-ui';c.fillText('−24h now +96h → tide height '+unit,8,12);
}
function liquidity(h){ let s=0; for(const m of MARKETS) if(sessOpen(m,h)) s+= m.n==='London'||m.n==='New York'?1:0.7; return s; }

const WATCHLIST=['SPY','NASDAQ','S&P500','BTC','XRP','SONY','MSFT','GOOG','SpaceX*'];
function spark(seed){ let v=50,r=seed; const pts=[]; for(let i=0;i<40;i++){ r=(r*1103515245+12345)%2147483648; v+=(r/2147483648-0.5)*8+(liquidity((utcHours(new Date())+i)%24)-1.2)*1.5; pts.push(v);} return pts; }
function renderWatch(){
  const el=document.getElementById('watch');el.innerHTML='';
  WATCHLIST.forEach((s,i)=>{ const d=document.createElement('div');d.className='sym';d.innerHTML=`<b>${s}</b> <span style="color:#5ef2b8">≈tide</span><canvas width="150" height="28"></canvas>`;
    const c=d.querySelector('canvas').getContext('2d');const p=spark(i*777+13);c.strokeStyle='#5ef2b8';c.beginPath();p.forEach((v,k)=>{const x=k/39*150,y=26-(v-30)/50*24;k?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();el.appendChild(d);});
}

function tick(){
  const now=new Date();
  draw(now);
  document.getElementById('bigTime').textContent=now.toISOString().slice(11,19)+' UTC';
  const cal=state.cal;
  document.getElementById('bigDate').textContent=calStr(now,cal)+` · ☀ ${sunLon(now).toFixed(1)}° δ${sunDecl(now).toFixed(1)}° · ☾ age ${moonAge(now).toFixed(1)}d illum ${(moonIllum(now)*100).toFixed(0)}%`;
  document.getElementById('altTime').textContent=`17h day: ${fmt17(now)} · Julian: ${julianDate(now).toISOString().slice(0,10)} · JDN ~${Math.floor(now.getTime()/864e5+2440588)}`;
  document.getElementById('calOut').innerHTML=`<b>${calStr(now,state.cal)}</b><br><span style="color:#8ea0c8">${['gregory','islamic','hebrew','chinese'].map(c=>{try{return c+': '+new Intl.DateTimeFormat('en-u-ca-'+c,{date:'short'}).format(now)}catch{return ''}}).join(' · ')}</span>`;
  // tides
  const ht=tideModel(now);
  const nextHi=[...Array(96)].map((_,i)=>new Date(now.getTime()+i*15*60e3)).find(d=>tideModel(d)>1.25);
  document.getElementById('tideNow').innerHTML=`Height (lat 33°): <b>${ht>=0?'+':''}${ht.toFixed(2)} m</b> ${ht>0.5?'▲ flood — tide coming in':'▼ ebb — tide going out'} · next high ≈ ${nextHi?nextHi.toISOString().slice(11,16)+' UTC':'—'} · springs near new/full moon`;
  drawSeries(document.getElementById('tideChart'),d=>tideModel(d),'#7fb2ff','m');
  const h=utcHours(now);
  const liq=liquidity(h);
  document.getElementById('liqNow').innerHTML=`Liquidity: <b>${'◉'.repeat(Math.round(liq))||'○'} ${liq.toFixed(1)}/3.4</b> · ${liq>1.5?'tide IN — oceans of wealth on the superhighway':'tide OUT — pools draining'} · Mon pour-in / Fri bail-out`;
  drawSeries(document.getElementById('liqChart'),d=>liquidity(utcHours(d))-1.2,'#5ef2b8','liquidity');
  const se=document.getElementById('sessions');se.innerHTML='';
  for(const m of MARKETS){ const o=sessOpen(m,h); const div=document.createElement('div');div.className='sess'+(o?' open':'');div.innerHTML=`<b>${m.n}</b> ${o?'OPEN':'closed'}<br><span style="color:#8ea0c8">${m.open}:00–${m.close}:00 UTC · ${(m.lon>0?'+':'')+m.lon}°</span>`;div.style.cursor='pointer';div.onclick=()=>{state.rot=-m.lon;};se.appendChild(div);}
  // holidays: show today's month-day + movables + pop
  const md=String(now.getUTCMonth()+1).padStart(2,'0')+'-'+String(now.getUTCDate()).padStart(2,'0');
  const el=document.getElementById('holidays');el.innerHTML='';
  HOLIDAYS.forEach(([k,v])=>{ const d=document.createElement('div');const today=k===md;d.className='hol'+(today?' today':'');d.innerHTML=`<b>${k}</b> — ${v}${today?' ◀ TODAY':''}`;el.appendChild(d);});
  const dayN=Math.floor(now/864e5);
  const q=QUOTES[dayN%QUOTES.length],w=WORDS[dayN%WORDS.length],k=KNOTS[dayN%KNOTS.length];
  document.getElementById('storyOut').innerHTML=`Today in story-time: <b>${HOLIDAYS[dayN%HOLIDAYS.length][1]}</b> · Mythic layer ${state.showMyth?'ON — Atlantis · Hyperborea · Meru + four rivers':'OFF'} · <span style="color:#8ea0c8">Groundhog Day sits next to Mt Zion on the same ring — compare, don't rank.</span>`;
  document.getElementById('dailyOut').innerHTML=`❝ ${q[1]} ❞ — <b>${q[0]}</b><br>Word-a-day: <b>${w[1]}</b> [${w[0]}] = ${w[2]} <span style="color:#8ea0c8">(week → usable sentence)</span><br>Knot-a-day: <b>${k}</b>`;
  document.getElementById('footDate').textContent=new Date().toUTCString();
}

// ---------- events ----------
function bind(id,fn){document.getElementById(id).onclick=e=>{fn(e);[...e.target.parentElement.children].forEach(b=>b.classList&&b.classList.remove('on'));if(e.target.tagName==='BUTTON')e.target.classList.add('on');};}
bind('modeClock',()=>state.mode='clock');bind('modeSky',()=>state.mode='sky');
bind('day24',()=>state.day=24);bind('day17',()=>state.day=17);
document.getElementById('calSel').onchange=e=>state.cal=e.target.value;
[['tNight','showNight'],['tMyth','showMyth'],['tMarkets','showMarkets'],['tLat33','showLat33']].forEach(([id,k])=>document.getElementById(id).onclick=e=>{state[k]=!state[k];e.target.classList.toggle('on',state[k]);});
let drag=null;
cv.addEventListener('pointerdown',e=>{drag={x:e.clientX,r:state.rot};cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointermove',e=>{if(drag)state.rot=drag.r+(e.clientX-drag.x)*0.2;});
cv.addEventListener('pointerup',()=>drag=null);
cv.addEventListener('wheel',e=>{e.preventDefault();state.zoom=Math.min(2.2,Math.max(.6,state.zoom*(e.deltaY<0?1.06:0.94)));},{passive:false});
renderWatch();setInterval(renderWatch,30000);
tick();setInterval(tick,1000);
