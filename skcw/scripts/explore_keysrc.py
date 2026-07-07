import paramiko, os

KEY = "C:/Users/94850/Desktop/id_rsa"
HOST, PORT = "xj-member.bitahub.com", 42166
BASE = "/LTX/ltx2.3/Semantic-STRR"
CW = f"{BASE}/condition_watermark"
R = f"{CW}/results"

key = paramiko.RSAKey.from_private_key_file(KEY)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username="root", pkey=key, allow_agent=False, look_for_keys=False)

cmds = [
    f"ls -la {R}/blind_key_id/",
    f"ls -la {R}/eval_hv10_semantic_mb512/",
    f"ls -la {R}/eval_hv10_semantic/",
    # heldout eval outputs
    f"find {R} -name '*heldout*' -o -name '*held_out*' 2>/dev/null | head -20",
    f"cat {CW}/heldout_eval_v6_last.log 2>/dev/null | tail -40",
    # peek blind_key_id json head
    f"for f in {R}/blind_key_id/*.json; do echo \"== $f ==\"; head -c 400 $f; echo; done 2>/dev/null | head -60",
    # semantic_summary mb512 structure (the SKCW one)
    f"python3 -c \"import json; d=json.load(open('{R}/eval_hv10_semantic_mb512/semantic_summary.json')); print(type(d)); print(list(d.keys()) if isinstance(d,dict) else len(d))\" 2>/dev/null",
]
for cmd in cmds:
    _, out, err = client.exec_command(cmd, timeout=40)
    result = out.read().decode().strip()
    e = err.read().decode().strip()
    print(f"\n>>> {cmd[:90]}")
    print((result or e)[:1800])

client.close()
