import { readFileSync, unlinkSync, existsSync } from "node:fs";
import { execSync } from "node:child_process";
import { join } from "node:path";

function stopProcessOnPort(port) {
  try {
    const pids = execSync(`lsof -tiTCP:${port} -sTCP:LISTEN`, {
      encoding: "utf8",
    })
      .trim()
      .split("\n")
      .filter(Boolean);

    for (const pid of pids) {
      console.log(`Stopping process on port ${port} (PID ${pid})...`);
      try {
        process.kill(Number(pid), "SIGTERM");
      } catch {
        // process may have already exited
      }
    }
  } catch {
    // no listener on this port
  }
}

function clearNextDevLock() {
  const lockPath = join(process.cwd(), ".next/dev/lock");

  if (!existsSync(lockPath)) {
    return;
  }

  try {
    const { pid } = JSON.parse(readFileSync(lockPath, "utf8"));

    try {
      process.kill(pid, 0);
      console.log(`Stopping existing Next.js dev server (PID ${pid})...`);
      process.kill(pid, "SIGTERM");
    } catch {
      unlinkSync(lockPath);
    }
  } catch {
    try {
      unlinkSync(lockPath);
    } catch {
      // ignore
    }
  }
}

stopProcessOnPort(8001);
clearNextDevLock();
