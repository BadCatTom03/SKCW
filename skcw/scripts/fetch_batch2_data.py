import paramiko, os

KEY = "C:/Users/94850/Desktop/id_rsa"
HOST, PORT = "xj-member.bitahub.com", 42166
BASE = "/LTX/ltx2.3/Semantic-STRR"
CW = f"{BASE}/condition_watermark"
R = f"{CW}/results"
OUT = "C:/Users/94850/Desktop/github开源代码/skcw/scripts/fetched"
os.makedirs(OUT, exist_ok=True)

fetch = {
    "blind_summary.json": f"{R}/blind_key_id/blind_summary.json",
    "skcw_mb512_semantic_summary.json": f"{R}/eval_hv10_semantic_mb512/semantic_summary.json",
    "skcw_mb512_semantic_results.jsonl": f"{R}/eval_hv10_semantic_mb512/semantic_results.jsonl",
    "skcw_mb32_semantic_summary.json": f"{R}/eval_hv10_semantic/semantic_summary.json",
    "skcw_mb32_semantic_results.jsonl": f"{R}/eval_hv10_semantic/semantic_results.jsonl",
    "heldout_eval_v6_last.json": f"{R}/e_theta_b2/heldout_eval_v6_last.json",
    "heldout_eval_v6.json": f"{R}/e_theta_b2/heldout_eval_v6.json",
}

key = paramiko.RSAKey.from_private_key_file(KEY)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username="root", pkey=key, allow_agent=False, look_for_keys=False)
sftp = client.open_sftp()
for fname, rp in fetch.items():
    try:
        sftp.get(rp, os.path.join(OUT, fname))
        print(f"[OK] {fname} ({os.path.getsize(os.path.join(OUT,fname))} bytes)")
    except Exception as e:
        print(f"[FAIL] {fname}: {e}")
sftp.close()

# Also dump full blind_summary so we can design the FPR script
_, out, _ = client.exec_command(f"cat {R}/blind_key_id/blind_summary.json", timeout=30)
print("\n===== blind_summary.json FULL =====")
print(out.read().decode()[:3000])
client.close()
