'use strict';
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const net = require('node:net');
const { spawn } = require('node:child_process');
const os = require('node:os');
const connectionPath = process.env.OMNISSIAH_PUBLISH_CONFIG || path.join(os.homedir(), '.config', 'omnissiahcore', 'publish.json');
const connection = fs.existsSync(connectionPath) ? JSON.parse(fs.readFileSync(connectionPath, 'utf8').replace(/^\uFEFF/, '')) : {};
const port = Number(process.env.ORACLE_TEST_PORT || 18001);
const upstreamPort = Number(process.env.ORACLE_TUNNEL_PORT || 18000);
const autoTunnel = process.env.ORACLE_SSH_AUTOSTART === '1';
const sshHost = process.env.ORACLE_SSH_HOST || connection.ssh_host;
const sshKey = process.env.ORACLE_SSH_KEY || connection.identity_file;
if (autoTunnel && (!sshHost || !sshKey)) {
  throw new Error('Configure the private publish.json file before starting an Oracle tunnel. See README.md.');
}
let tunnel = null;
let stopping = false;
let checkingTunnel = false;
function tunnelPortOpen() {
  return new Promise(resolve => {
    const socket = net.connect({ host: '127.0.0.1', port: upstreamPort });
    const finish = open => { socket.destroy(); resolve(open); };
    socket.once('connect', () => finish(true));
    socket.once('error', () => finish(false));
    socket.setTimeout(1200, () => finish(false));
  });
}
async function ensureTunnel() {
  if (!autoTunnel || tunnel || stopping || checkingTunnel) return;
  checkingTunnel = true;
  try {
    if (await tunnelPortOpen() || stopping) return;
    tunnel = spawn('ssh', [
      '-i', sshKey, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
      '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30',
      '-o', 'ServerAliveCountMax=3', '-N', '-L',
      '127.0.0.1:' + upstreamPort + ':127.0.0.1:8000', sshHost
    ], { windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    tunnel.stderr.on('data', chunk => { stderr = (stderr + chunk.toString()).slice(-500); });
    tunnel.once('error', error => { console.error('SSH could not start:', error.message); tunnel = null; });
    tunnel.once('exit', code => {
      if (!stopping) console.error('SSH tunnel stopped (exit ' + code + '). ' + stderr.trim());
      tunnel = null;
    });
    console.log('Starting SSH tunnel to Oracle on port ' + upstreamPort);
  } finally { checkingTunnel = false; }
}
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
if (autoTunnel) {
  ensureTunnel();
  setInterval(ensureTunnel, 5000);
}
function shutdown() {
  stopping = true;
  if (tunnel) tunnel.kill();
  server.close(() => process.exit(0));
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
