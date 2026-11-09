// Minimal bridge so the existing Node bots can submit experiments and read
// results without any new npm dependencies. Uses the qrlab JSON CLI.
//
//   const qrlab = require('./quant_research/integrations/node/qrlab');
//   const job = await qrlab.submit({ name: 'btc-7d', domain: 'crypto', engine: 'optuna',
//                                    data: { type: 'synthetic' }, horizon: 7 }, 'grokbot-crypto');
//   const res = await qrlab.result(job.id, { wait: true });
const { execFile } = require('child_process');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const PY = process.env.QR_CORE_PYTHON || path.join(ROOT, '.venvs', 'core', 'bin', 'python');

function cli(args, stdin) {
  return new Promise((resolve, reject) => {
    const child = execFile(PY, ['-m', 'qrlab', ...args], { cwd: ROOT, maxBuffer: 64 * 1024 * 1024 },
      (err, stdout, stderr) => {
        if (err) return reject(new Error(stderr || err.message));
        try { resolve(JSON.parse(stdout)); } catch (e) { reject(new Error(`bad JSON: ${stdout}`)); }
      });
    if (stdin) { child.stdin.write(JSON.stringify(stdin)); child.stdin.end(); }
  });
}

module.exports = {
  resources: () => cli(['resources']),
  estimate: spec => cli(['estimate', '-'], spec),
  submit: (spec, agent = 'node-bot') => cli(['submit', '-', '--agent', agent], spec),
  status: id => cli(['status', id]),
  result: (id, { wait = false, timeout = 3600 } = {}) =>
    cli(['result', id, ...(wait ? ['--wait', '--timeout', String(timeout)] : [])]),
  history: domain => cli(['history', ...(domain ? ['--domain', domain] : [])]),
};
