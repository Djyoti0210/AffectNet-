import subprocess, time

vals_util = []
vals_mem = []
print("Sampling GPU...")
for _ in range(10):
    try:
        res = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits']).decode('utf-8')
        util, mem = res.strip().split(', ')
        vals_util.append(int(util))
        vals_mem.append(int(mem))
    except:
        pass
    time.sleep(3)

print(f"GPU Util: Min {min(vals_util)}%, Max {max(vals_util)}%, Avg {sum(vals_util)/len(vals_util):.1f}%")
print(f"VRAM: Min {min(vals_mem)}MB, Max {max(vals_mem)}MB, Avg {sum(vals_mem)/len(vals_mem):.1f}MB")
