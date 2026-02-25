"""
CPU Scheduling Simulation Project
Simulates 5 scheduling algorithms with 20 processes and a transient event.
Algorithms: FCFS, SJF, Round Robin, Priority Scheduling, SRTF
"""

import random
import json
import copy

random.seed(42)

# ── Process Generation ──────────────────────────────────────────────────────
NUM_PROCESSES = 20

def generate_processes(n):
    """Generate n processes with arrival_time, burst_time, and priority."""
    processes = []
    for i in range(1, n + 1):
        processes.append({
            'pid': f'P{i}',
            'arrival_time': random.randint(0, 30),
            'burst_time': random.randint(1, 20),
            'priority': random.randint(1, 10),  # 1 = highest priority
        })
    # Sort by arrival time for readability
    processes.sort(key=lambda p: (p['arrival_time'], p['pid']))
    return processes

processes = generate_processes(NUM_PROCESSES)

# ── Transient Event ─────────────────────────────────────────────────────────
# At time = 15, a sudden I/O interrupt causes 3 new high-priority,
# short-burst processes to arrive simultaneously (simulating an interrupt storm).
TRANSIENT_TIME = 15
TRANSIENT_PROCESSES = [
    {'pid': 'T1', 'arrival_time': TRANSIENT_TIME, 'burst_time': 2, 'priority': 1},
    {'pid': 'T2', 'arrival_time': TRANSIENT_TIME, 'burst_time': 3, 'priority': 1},
    {'pid': 'T3', 'arrival_time': TRANSIENT_TIME, 'burst_time': 1, 'priority': 2},
]

def get_all_processes(base, transient):
    """Merge base processes with transient event processes."""
    combined = copy.deepcopy(base) + copy.deepcopy(transient)
    combined.sort(key=lambda p: (p['arrival_time'], p['pid']))
    return combined

# ── Helper: compute metrics ─────────────────────────────────────────────────
def compute_metrics(schedule):
    """Given a list of {'pid', 'start', 'end', 'arrival_time', 'burst_time'},
       compute per-process and aggregate metrics."""
    # Group by pid
    pid_data = {}
    for entry in schedule:
        pid = entry['pid']
        if pid not in pid_data:
            pid_data[pid] = {
                'pid': pid,
                'arrival_time': entry['arrival_time'],
                'burst_time': entry['burst_time'],
                'priority': entry.get('priority', 0),
                'first_start': entry['start'],
                'completion_time': entry['end'],
                'segments': []
            }
        else:
            pid_data[pid]['completion_time'] = max(pid_data[pid]['completion_time'], entry['end'])
            pid_data[pid]['first_start'] = min(pid_data[pid]['first_start'], entry['start'])
        pid_data[pid]['segments'].append((entry['start'], entry['end']))

    results = []
    for pid, d in pid_data.items():
        ct = d['completion_time']
        at = d['arrival_time']
        bt = d['burst_time']
        tat = ct - at
        wt = tat - bt
        rt = d['first_start'] - at
        results.append({
            'pid': pid,
            'arrival_time': at,
            'burst_time': bt,
            'priority': d['priority'],
            'completion_time': ct,
            'turnaround_time': tat,
            'waiting_time': wt,
            'response_time': rt,
        })

    results.sort(key=lambda x: x['pid'])
    n = len(results)
    avg_tat = sum(r['turnaround_time'] for r in results) / n
    avg_wt = sum(r['waiting_time'] for r in results) / n
    avg_rt = sum(r['response_time'] for r in results) / n
    throughput = n / max(r['completion_time'] for r in results) if results else 0

    return results, {
        'avg_turnaround_time': round(avg_tat, 2),
        'avg_waiting_time': round(avg_wt, 2),
        'avg_response_time': round(avg_rt, 2),
        'throughput': round(throughput, 4),
        'total_completion_time': max(r['completion_time'] for r in results),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULING ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. FCFS (First Come First Serve) ────────────────────────────────────────
def fcfs(procs):
    procs = sorted(copy.deepcopy(procs), key=lambda p: (p['arrival_time'], p['pid']))
    schedule = []
    current_time = 0
    for p in procs:
        if current_time < p['arrival_time']:
            current_time = p['arrival_time']
        start = current_time
        end = start + p['burst_time']
        schedule.append({
            'pid': p['pid'], 'start': start, 'end': end,
            'arrival_time': p['arrival_time'], 'burst_time': p['burst_time'],
            'priority': p.get('priority', 0)
        })
        current_time = end
    return schedule


# ── 2. SJF (Shortest Job First - Non-Preemptive) ────────────────────────────
def sjf(procs):
    procs = copy.deepcopy(procs)
    schedule = []
    current_time = 0
    remaining = sorted(procs, key=lambda p: (p['arrival_time'], p['burst_time']))
    completed = set()

    while len(completed) < len(procs):
        available = [p for p in remaining if p['pid'] not in completed and p['arrival_time'] <= current_time]
        if not available:
            # Jump to next arrival
            future = [p for p in remaining if p['pid'] not in completed]
            current_time = min(p['arrival_time'] for p in future)
            continue
        # Pick shortest burst
        chosen = min(available, key=lambda p: (p['burst_time'], p['arrival_time']))
        start = current_time
        end = start + chosen['burst_time']
        schedule.append({
            'pid': chosen['pid'], 'start': start, 'end': end,
            'arrival_time': chosen['arrival_time'], 'burst_time': chosen['burst_time'],
            'priority': chosen.get('priority', 0)
        })
        current_time = end
        completed.add(chosen['pid'])
    return schedule


# ── 3. Round Robin ──────────────────────────────────────────────────────────
def round_robin(procs, quantum=4):
    procs = sorted(copy.deepcopy(procs), key=lambda p: (p['arrival_time'], p['pid']))
    schedule = []
    current_time = 0
    queue = []
    remaining_bt = {p['pid']: p['burst_time'] for p in procs}
    proc_map = {p['pid']: p for p in procs}
    arrived = set()
    completed = set()
    idx = 0  # index into sorted procs

    def enqueue_new_arrivals(up_to_time):
        nonlocal idx
        while idx < len(procs) and procs[idx]['arrival_time'] <= up_to_time:
            pid = procs[idx]['pid']
            if pid not in arrived:
                queue.append(pid)
                arrived.add(pid)
            idx += 1

    enqueue_new_arrivals(current_time)

    while len(completed) < len(procs):
        if not queue:
            # Jump to next arrival
            future = [p for p in procs if p['pid'] not in completed and p['pid'] not in arrived]
            if not future:
                break
            current_time = future[0]['arrival_time']
            enqueue_new_arrivals(current_time)
            continue

        pid = queue.pop(0)
        p = proc_map[pid]
        exec_time = min(quantum, remaining_bt[pid])
        start = current_time
        end = start + exec_time
        schedule.append({
            'pid': pid, 'start': start, 'end': end,
            'arrival_time': p['arrival_time'], 'burst_time': p['burst_time'],
            'priority': p.get('priority', 0)
        })
        current_time = end
        remaining_bt[pid] -= exec_time

        # Enqueue new arrivals that came during execution
        enqueue_new_arrivals(current_time)

        if remaining_bt[pid] > 0:
            queue.append(pid)
        else:
            completed.add(pid)

    return schedule


# ── 4. Priority Scheduling (Non-Preemptive, lower number = higher priority) ─
def priority_scheduling(procs):
    procs = copy.deepcopy(procs)
    schedule = []
    current_time = 0
    completed = set()

    while len(completed) < len(procs):
        available = [p for p in procs if p['pid'] not in completed and p['arrival_time'] <= current_time]
        if not available:
            future = [p for p in procs if p['pid'] not in completed]
            current_time = min(p['arrival_time'] for p in future)
            continue
        # Pick highest priority (lowest number), break ties with arrival time
        chosen = min(available, key=lambda p: (p['priority'], p['arrival_time']))
        start = current_time
        end = start + chosen['burst_time']
        schedule.append({
            'pid': chosen['pid'], 'start': start, 'end': end,
            'arrival_time': chosen['arrival_time'], 'burst_time': chosen['burst_time'],
            'priority': chosen.get('priority', 0)
        })
        current_time = end
        completed.add(chosen['pid'])
    return schedule


# ── 5. SRTF (Shortest Remaining Time First - Preemptive SJF) ────────────────
def srtf(procs):
    procs = copy.deepcopy(procs)
    schedule = []
    current_time = 0
    remaining_bt = {p['pid']: p['burst_time'] for p in procs}
    proc_map = {p['pid']: p for p in procs}
    completed = set()
    n = len(procs)

    # Get all unique event times (arrivals)
    all_arrivals = sorted(set(p['arrival_time'] for p in procs))

    current_pid = None
    seg_start = 0

    while len(completed) < n:
        available = [p for p in procs if p['pid'] not in completed and p['arrival_time'] <= current_time and remaining_bt[p['pid']] > 0]

        if not available:
            future = [p for p in procs if p['pid'] not in completed]
            if not future:
                break
            current_time = min(p['arrival_time'] for p in future)
            continue

        chosen = min(available, key=lambda p: (remaining_bt[p['pid']], p['arrival_time']))

        # Find next event: either next arrival or completion of chosen
        next_arrivals = [p['arrival_time'] for p in procs if p['arrival_time'] > current_time and p['pid'] not in completed]
        time_to_complete = remaining_bt[chosen['pid']]
        next_event = current_time + time_to_complete

        if next_arrivals:
            nearest_arrival = min(next_arrivals)
            if nearest_arrival < next_event:
                next_event = nearest_arrival

        exec_time = next_event - current_time
        start = current_time
        end = start + exec_time

        schedule.append({
            'pid': chosen['pid'], 'start': start, 'end': end,
            'arrival_time': chosen['arrival_time'], 'burst_time': chosen['burst_time'],
            'priority': chosen.get('priority', 0)
        })

        remaining_bt[chosen['pid']] -= exec_time
        current_time = end

        if remaining_bt[chosen['pid']] <= 0:
            completed.add(chosen['pid'])

    return schedule


# ══════════════════════════════════════════════════════════════════════════════
# RUN SIMULATIONS
# ══════════════════════════════════════════════════════════════════════════════

# --- Without transient event ---
results_no_transient = {}
all_procs_base = copy.deepcopy(processes)

algorithms_base = {
    'FCFS': fcfs(all_procs_base),
    'SJF': sjf(all_procs_base),
    'Round Robin (Q=4)': round_robin(all_procs_base, quantum=4),
    'Priority': priority_scheduling(all_procs_base),
    'SRTF': srtf(all_procs_base),
}

for name, sched in algorithms_base.items():
    per_proc, agg = compute_metrics(sched)
    results_no_transient[name] = {'per_process': per_proc, 'aggregate': agg, 'schedule': sched}

# --- With transient event ---
results_with_transient = {}
all_procs_transient = get_all_processes(processes, TRANSIENT_PROCESSES)

algorithms_transient = {
    'FCFS': fcfs(all_procs_transient),
    'SJF': sjf(all_procs_transient),
    'Round Robin (Q=4)': round_robin(all_procs_transient, quantum=4),
    'Priority': priority_scheduling(all_procs_transient),
    'SRTF': srtf(all_procs_transient),
}

for name, sched in algorithms_transient.items():
    per_proc, agg = compute_metrics(sched)
    results_with_transient[name] = {'per_process': per_proc, 'aggregate': agg, 'schedule': sched}


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("PROCESS TABLE")
print("=" * 80)
print(f"{'PID':<6} {'Arrival':<10} {'Burst':<8} {'Priority':<10}")
print("-" * 40)
for p in processes:
    print(f"{p['pid']:<6} {p['arrival_time']:<10} {p['burst_time']:<8} {p['priority']:<10}")

print(f"\nTransient Event at t={TRANSIENT_TIME}: 3 new processes arrive")
for p in TRANSIENT_PROCESSES:
    print(f"  {p['pid']}: burst={p['burst_time']}, priority={p['priority']}")

print("\n" + "=" * 80)
print("RESULTS WITHOUT TRANSIENT EVENT")
print("=" * 80)
for name, data in results_no_transient.items():
    agg = data['aggregate']
    print(f"\n--- {name} ---")
    print(f"  Avg Turnaround Time: {agg['avg_turnaround_time']}")
    print(f"  Avg Waiting Time:    {agg['avg_waiting_time']}")
    print(f"  Avg Response Time:   {agg['avg_response_time']}")
    print(f"  Throughput:          {agg['throughput']}")
    print(f"  Total Completion:    {agg['total_completion_time']}")

print("\n" + "=" * 80)
print("RESULTS WITH TRANSIENT EVENT")
print("=" * 80)
for name, data in results_with_transient.items():
    agg = data['aggregate']
    print(f"\n--- {name} ---")
    print(f"  Avg Turnaround Time: {agg['avg_turnaround_time']}")
    print(f"  Avg Waiting Time:    {agg['avg_waiting_time']}")
    print(f"  Avg Response Time:   {agg['avg_response_time']}")
    print(f"  Throughput:          {agg['throughput']}")
    print(f"  Total Completion:    {agg['total_completion_time']}")

# ── Transient-only process metrics ──────────────────────────────────────────
print("\n" + "=" * 80)
print("TRANSIENT PROCESS RESPONSE TIMES (T1, T2, T3)")
print("=" * 80)
for name, data in results_with_transient.items():
    transient_procs = [r for r in data['per_process'] if r['pid'].startswith('T')]
    if transient_procs:
        avg_rt = sum(r['response_time'] for r in transient_procs) / len(transient_procs)
        avg_wt = sum(r['waiting_time'] for r in transient_procs) / len(transient_procs)
        print(f"\n--- {name} ---")
        for tp in transient_procs:
            print(f"  {tp['pid']}: Response={tp['response_time']}, Waiting={tp['waiting_time']}, TAT={tp['turnaround_time']}")
        print(f"  Avg Response Time (transient): {avg_rt:.2f}")
        print(f"  Avg Waiting Time (transient):  {avg_wt:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE DATA FOR CHARTS AND REPORT
# ══════════════════════════════════════════════════════════════════════════════
output = {
    'processes': processes,
    'transient_processes': TRANSIENT_PROCESSES,
    'transient_time': TRANSIENT_TIME,
    'results_no_transient': {},
    'results_with_transient': {},
}

for name in results_no_transient:
    output['results_no_transient'][name] = {
        'per_process': results_no_transient[name]['per_process'],
        'aggregate': results_no_transient[name]['aggregate'],
        'schedule': results_no_transient[name]['schedule'],
    }
    output['results_with_transient'][name] = {
        'per_process': results_with_transient[name]['per_process'],
        'aggregate': results_with_transient[name]['aggregate'],
        'schedule': results_with_transient[name]['schedule'],
    }

with open('/home/claude/sim_data.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✓ Simulation data saved to sim_data.json")
