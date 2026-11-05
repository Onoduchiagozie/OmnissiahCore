'use strict';
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const port = Number(process.env.ORACLE_TEST_PORT || 18001);
const upstreamPort = Number(process.env.ORACLE_TUNNEL_PORT || 18000);
const allowed = new Set(['/query/focus','/query/explore','/query/narrate','/query/focus/stream','/query/inspect','/query/ghost','/health']);
const html = fs.readFileSync(path.join(__dirname,'index.html'));
const server = http.createServer((req,res)=>{
  if(req.method==='GET' && (req.url==='/' || req.url==='/index.html')){
    res.writeHead(200,{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store'});
    return res.end(html);
  }
  if(req.url==='/favicon.ico'){res.writeHead(204);return res.end();}
  const route = (req.url || '').startsWith('/api/') ? req.url.slice(4) : '';
  if(!allowed.has(route) || (route==='/health'?req.method!=='GET':req.method!=='POST')){
    res.writeHead(404,{'Content-Type':'application/json'});return res.end(JSON.stringify({proxy_error:'Unknown test route'}));
  }
  if(req.headers.origin && !['http://127.0.0.1:'+port,'http://localhost:'+port].includes(req.headers.origin)){
    res.writeHead(403);return res.end('Origin rejected');
  }
  const upstream=http.request({hostname:'127.0.0.1',port:upstreamPort,path:route,method:req.method,
    headers:{'Content-Type':'application/json'}},response=>{
      res.writeHead(response.statusCode,{'Content-Type':response.headers['content-type']||'application/json',
        'Cache-Control':'no-store','X-Oracle-Route':route});
      response.pipe(res);
  });
  upstream.setTimeout(300000,()=>upstream.destroy(new Error('Oracle response timed out')));
  upstream.on('error',error=>{
    if(!res.headersSent)res.writeHead(502,{'Content-Type':'application/json'});
    if(!res.writableEnded)res.end(JSON.stringify({proxy_error:'Oracle connection failed: '+error.message}));
  });
  res.on('close',()=>{if(!res.writableEnded)upstream.destroy();});
  req.pipe(upstream);
});
server.listen(port,'127.0.0.1',()=>console.log('LIVE Oracle test console: http://127.0.0.1:'+port+' (SSH tunnel upstream port '+upstreamPort+')'));
server.on('error',error=>{console.error(error.message);process.exitCode=1;});
