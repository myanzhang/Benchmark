import os

f = os.popen("ps -ef | grep  dlrm | grep -v grep | awk '{print $2}'")

cpu_id = 1
for pid in f:
    r = os.popen("taskset -pc {} {}".format(cpu_id, pid).strip())
    print("Successfully set cpu {} affinity to PID {}".format(cpu_id, pid.strip()))
    cpu_id += 1
