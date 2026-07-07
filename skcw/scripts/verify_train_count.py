import paramiko

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
    # Count training latents (d1_latents = E_theta training data)
    f"ls {R}/d1_latents/ | grep -c npz",
    f"ls {R}/d1_latents_hv10/ | grep -c npz",
    f"ls {R}/d1_latents/ | head -5",
    # E_theta training config — how many samples/augmentations
    f"cat {R}/e_theta_b2/heldout_eval_v6.json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(\"batch\",d.get(\"batch\"),\"n_batches\",d.get(\"n_batches\"))'",
    # look for train manifest / dataset size in train report or config
    f"find {R}/e_theta_b2 -name '*.json' 2>/dev/null",
    f"find {R}/e_theta_hv10 -name '*.json' 2>/dev/null | head",
    # training log: grep for dataset size / num samples / slices
    f"grep -iE 'train.*(sample|latent|slice|video|size)|dataset.*[0-9]' {CW}/m3_train_v6.log 2>/dev/null | head -20",
    # how many videos generated for training (class 0-27)
    f"find {R}/d1_latents -name 'c*.npz' 2>/dev/null | sed 's/.*\\/c\\([0-9]*\\)_.*/\\1/' | sort -un | tr '\\n' ' '",
    # check npz content (one latent = how many temporal slices?)
    f"python3 -c \"import numpy as np; d=np.load('{R}/d1_latents/c00_s0.npz'); print({{k:d[k].shape for k in d.files}})\" 2>/dev/null || echo NO_C00",
    f"ls {R}/d1_latents/ | grep npz | wc -l",
]
for cmd in cmds:
    _, out, err = client.exec_command(cmd, timeout=40)
    res = out.read().decode().strip()
    e = err.read().decode().strip()
    print(f"\n>>> {cmd[:80]}")
    print((res or e)[:600])

client.close()
