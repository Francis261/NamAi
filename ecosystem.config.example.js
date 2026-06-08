// Copy to ecosystem.config.js and set your HF_TOKEN
module.exports = {
  apps: [{
    name: "mamba-training",
    script: "/usr/bin/python3",
    args: [
      "-m", "src.training.train",
      "--config", "configs/tiny_cpu.yaml",
      "--data_mix", "configs/data_mix_cpu.yaml",
      "--save_dir", "checkpoints",
      "--batch_size", "1",
      "--grad_accum", "1",
      "--log_interval", "1",
      "--save_interval", "50",
      "--no_auto_resume",
    ],
    cwd: "/home/admin/aii",
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    error_file: "logs/train-err.log",
    out_file: "logs/train-out.log",
    merge_logs: true,
    max_restarts: 10,
    restart_delay: 5000,
    autorestart: true,
    kill_timeout: 30000,
    env: {
      PYTHONUNBUFFERED: "1",
      HF_TOKEN: "YOUR_HF_TOKEN_HERE",
    },
  }],
};
