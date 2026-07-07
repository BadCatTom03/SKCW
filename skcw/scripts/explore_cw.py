import paramiko, os

KEY = "C:/Users/94850/Desktop/id_rsa"
HOST, PORT = "xj-member.bitahub.com", 42166
BASE = "/LTX/ltx2.3/Semantic-STRR"
CW = f"{BASE}/condition_watermark"

key = paramiko.RSAKey.from_private_key_file(KEY)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username="root", pkey=key, allow_agent=False, look_for_keys=False)

cmds = [
    f"ls -la {CW}/results/",
    f"find {CW}/results -maxdepth 2 -type d 2>/dev/null",
    f"find {CW}/results -name '*.json' 2>/dev/null | head -60",
    f"find {CW}/results -name '*.npy' -o -name '*.npz' -o -name '*.pkl' 2>/dev/null | head -40",
    f"find {CW}/results -name '*null*' -o -name '*clean*' -o -name '*score*' 2>/dev/null | head -40",
    f"find {CW} -maxdepth 1 -type d 2>/dev/null",
    f"ls {CW}/ 2>/dev/null",
]
for cmd in cmds:
    _, out, err = client.exec_command(cmd, timeout=40)
    result = out.read().decode().strip()
    print(f"\n>>> {cmd[:90]}")
    print(result[:2500] if result else "(empty)")

client.close()
