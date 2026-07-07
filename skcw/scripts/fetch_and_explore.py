import paramiko, json, os

KEY = "C:/Users/94850/Desktop/id_rsa"
HOST, PORT = "xj-member.bitahub.com", 42166
BASE = "/LTX/ltx2.3/Semantic-STRR"
OUT = "C:/Users/94850/Desktop/github开源代码/skcw/scripts/fetched"

os.makedirs(OUT, exist_ok=True)

# Files to fetch
fetch_files = [
    f"{BASE}/SIGMark-release-main/outputs-hv-t2v/sigmark_semantic_512bit/semantic_summary.json",
    f"{BASE}/SIGMark-release-main/outputs-hv-t2v/videomark_semantic_512bit/semantic_summary.json",
    f"{BASE}/SIGMark-release-main/outputs-hv-t2v/videoshield_semantic_512bit/semantic_summary.json",
]

# Commands to discover SKCW score/detection files
explore_cmds = [
    f"ls {BASE}/outputs/ 2>/dev/null | head -30",
    f"ls {BASE}/test_robustness_output/ 2>/dev/null | head -30",
    f"find {BASE}/outputs -name '*.json' 2>/dev/null | head -30",
    f"find {BASE}/test_robustness_output -name '*.json' 2>/dev/null | head -30",
    f"find {BASE}/skcw_bernini -name '*.json' 2>/dev/null | head -20",
    # Look for score arrays / null scores saved by SKCW
    f"find {BASE} -name '*score*' -name '*.json' 2>/dev/null | grep -v venv | head -20",
    f"find {BASE} -name '*null*' -name '*.json' 2>/dev/null | grep -v venv | head -20",
    f"find {BASE} -name '*clean*' -name '*.json' 2>/dev/null | grep -v venv | head -20",
    f"find {BASE} -name '*detection*' -name '*.json' 2>/dev/null | grep -v venv | head -20",
    # Check semantic_key_orthogonality file (has cross-key scores!)
    f"cat {BASE}/semantic_key_orthogonality_prc8.json 2>/dev/null | head -100",
    f"cat {BASE}/semantic_key_robustness_p2.json 2>/dev/null | head -80",
    # Check main result files in outputs
    f"find {BASE}/outputs -maxdepth 3 -name '*.json' 2>/dev/null",
    f"find {BASE}/outputs -maxdepth 3 -name '*.pkl' 2>/dev/null | head -10",
    # training data count
    f"find {BASE} -path '*train*' -name '*.json' 2>/dev/null | grep -v venv | head -10",
    f"ls {BASE}/outputs/ 2>/dev/null",
]

key = paramiko.RSAKey.from_private_key_file(KEY)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username="root", pkey=key, allow_agent=False, look_for_keys=False)

# Fetch JSON files
sftp = client.open_sftp()
for remote_path in fetch_files:
    fname = os.path.basename(os.path.dirname(remote_path)) + "_" + os.path.basename(remote_path)
    local_path = os.path.join(OUT, fname)
    try:
        sftp.get(remote_path, local_path)
        print(f"[OK] {fname}")
    except Exception as e:
        print(f"[FAIL] {fname}: {e}")
sftp.close()

# Run explore commands
for cmd in explore_cmds:
    _, out, err = client.exec_command(cmd, timeout=30)
    result = out.read().decode().strip()
    print(f"\n>>> {cmd[:90]}")
    print(result[:600] if result else "(empty)")

client.close()
