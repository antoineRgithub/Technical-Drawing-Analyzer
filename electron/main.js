const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

let mainWindow;
let pyProc;

const logFile = path.join(app.getPath("userData"), "app.log");
function log(msg) {
    const line = `[${new Date().toISOString()}] ${msg}\n`;
    fs.appendFileSync(logFile, line);
    console.log(msg);
}

function getStreamlitPath() {
    if (app.isPackaged) {
        return path.join(process.resourcesPath, "app", "venv", "Scripts", "streamlit.exe");
    } else {
        return path.join(__dirname, "..", "venv", "Scripts", "streamlit.exe");
    }
}

function getAppPath() {
    if (app.isPackaged) {
        return path.join(process.resourcesPath, "app.py");
    } else {
        return path.join(__dirname, "..", "app.py");
    }
}
function createWindow() {
    mainWindow = new BrowserWindow({ width: 1200, height: 800 });
    mainWindow.loadURL("http://localhost:8501");
}

function startStreamlit() {
    const streamlitPath = getStreamlitPath();
    const appPath = getAppPath();

    log(`isPackaged: ${app.isPackaged}`);
    log(`streamlitPath: ${streamlitPath}`);
    log(`appPath: ${appPath}`);
    log(`streamlit exists: ${fs.existsSync(streamlitPath)}`);
    log(`app.py exists: ${fs.existsSync(appPath)}`);

    pyProc = spawn(streamlitPath, [
        "run",
        appPath,
        "--server.headless=true"
    ]);

    pyProc.stdout.on("data", (data) => log(`[streamlit stdout] ${data.toString()}`));
    pyProc.stderr.on("data", (data) => log(`[streamlit stderr] ${data.toString()}`));
    pyProc.on("error", (err) => log(`[streamlit error] ${JSON.stringify(err)}`));
    pyProc.on("close", (code) => log(`[streamlit closed] code: ${code}`));
}

app.whenReady().then(() => {
    log("App ready");
    startStreamlit();
    setTimeout(() => createWindow(), 5000);
});

app.on("will-quit", () => {
    if (pyProc) pyProc.kill();
});