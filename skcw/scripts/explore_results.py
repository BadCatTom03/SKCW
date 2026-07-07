import paramiko, json

KEY = "C:/Users/94850/Desktop/id_rsa"
HOST, PORT = "xj-member.bitahub.com", 42166
BASE = "/LTX/ltx2.3/Semantic-STRR"

cmds = [
    f"ls {BASE}/",
    f"ls {BASE}/results/ 2>/dev/null || echo NO_RESULTS",
    f"find {BASE} -name '*.json' -newer {BASE}/SIGMark-release-main 2>/dev/null | head -30",
    f"find {BASE} -name 'null_scores*' -o -name 'clean_scores*' -o -name '*detection*' 2>/dev/null | head -20",
    f"find {BASE} -name 'test_results*' -o -name 'eval_results*' 2>/dev/null | head -20",
    # STRR output structure
    f"ls {BASE}/STRR/ 2>/dev/null | head -20",
    f"find {BASE}/STRR -name '*.json' 2>/dev/null | head -30",
    # SIGMark results
    f"find {BASE}/SIGMark-release-main -name '*.json' 2>/dev/null | head -30",
    # check pkl files (maintained_info)
    f"find {BASE} -name 'maintained_info*' 2>/dev/null | head -10",
]

key = paramiko.RSAKey.from_private_key_file(KEY)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username="root", pkey=key, allow_agent=False, look_for_keys=False)

for cmd in cmds:
    _, out, err = client.exec_command(cmd)
    result = out.read().decode().strip()
    print(f"\n>>> {cmd[:80]}")
    print(result[:800] if result else err.read().decode().strip()[:200])

client.close()
