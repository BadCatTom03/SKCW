import paramiko, os

KEY = "C:/Users/94850/Desktop/id_rsa"
HOST, PORT = "xj-member.bitahub.com", 42166
BASE = "/LTX/ltx2.3/Semantic-STRR"
OUT = "C:/Users/94850/Desktop/github开源代码/skcw/scripts/fetched"
os.makedirs(OUT, exist_ok=True)

fetch_files = {
    "skcw_robustness_results.json": f"{BASE}/test_robustness_output/robustness_results.json",
    "skcw_semantic_bound_results.json": f"{BASE}/test_robustness_output/semantic_bound_results.json",
    "skcw_key_orthogonality_prc8.json": f"{BASE}/semantic_key_orthogonality_prc8.json",
    "skcw_key_robustness_p2.json": f"{BASE}/semantic_key_robustness_p2.json",
    "route_c_train_report.json": f"{BASE}/outputs/route_c_dataset/route_c_train_report.json",
    "route_c_render_report.json": f"{BASE}/outputs/route_c_dataset/route_c_render_report.json",
}

key = paramiko.RSAKey.from_private_key_file(KEY)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username="root", pkey=key, allow_agent=False, look_for_keys=False)
sftp = client.open_sftp()
for fname, rp in fetch_files.items():
    try:
        sftp.get(rp, os.path.join(OUT, fname))
        sz = os.path.getsize(os.path.join(OUT, fname))
        print(f"[OK] {fname} ({sz} bytes)")
    except Exception as e:
        print(f"[FAIL] {fname}: {e}")
sftp.close()
client.close()
