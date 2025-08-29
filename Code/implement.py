import numpy as np
import math
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import copy
import pandas as pd

# --- Helper Functions ---

def lcm(a, b):
  """Calculates the least common multiple of two integers."""
  return abs(a*b) // math.gcd(a, b) if a != 0 and b != 0 else 0

def calculate_hyperperiod(periods: List[int]) -> int:
  """Calculates the hyperperiod for a list of task periods."""
  if not periods:
    return 0
  result = periods[0]
  for i in range(1, len(periods)):
    result = lcm(result, periods[i])
  return result

def get_frames(periods: List[int], hyperperiod: int) -> List[Tuple[int, int]]:
  """Computes frames based on Deadline Partitioning."""
  if hyperperiod == 0:
      return []
  deadlines = set([0, hyperperiod])
  for p in periods:
      k = 1
      while k * p <= hyperperiod:
          deadlines.add(k * p)
          k += 1
  
  sorted_deadlines = sorted(list(deadlines))
  frames = []
  for i in range(len(sorted_deadlines) - 1):
      frames.append((sorted_deadlines[i], sorted_deadlines[i+1]))
  return frames

def calculate_uf(tasks, num_cores):
    """Calculates the Utilization Factor (UF)."""
    if not tasks or num_cores == 0:
        return 0
    total_avg_util = 0
    for task in tasks:
        # Use nominal (average) utilization for UF calculation
        nominal_utils = task.utilization_fixed if hasattr(task, 'utilization_fixed') else \
                       [(u_min + u_max) / 2 for u_min, u_max in task.utilization_elastic]
        if nominal_utils:
             total_avg_util += np.mean([u for u in nominal_utils if u is not None]) # Handle potential None values if a task can't run on a core
    return total_avg_util / num_cores

# --- Data Structures ---

@dataclass
class Task:
    id: int
    period: int
    # For COST
    utilization_fixed: Optional[List[Optional[float]]] = None
    # For COST-Elastic
    utilization_elastic: Optional[List[Tuple[Optional[float], Optional[float]]]] = None # List of (min_util, max_util) per core
    # Runtime assignment
    assigned_cluster_id: Optional[int] = None
    assigned_core_ids: List[int] = field(default_factory=list) # Can be one or two for migration
    current_utilization: Optional[List[Optional[float]]] = None # Util actually assigned by Elastic scheduler

    def get_elasticity_potential(self, num_cores) -> float:
        """Calculates elasticity potential for COST-Elastic."""
        if not self.utilization_elastic:
            return 0.0
        potential = 0.0
        for i in range(num_cores):
             min_u, max_u = self.utilization_elastic[i] if self.utilization_elastic[i] is not None else (None, None)
             if min_u is not None and max_u is not None:
                 potential += (max_u - min_u)
        return potential

    def get_nominal_utilization(self, core_index: int) -> Optional[float]:
       """Returns the nominal (mid-point) utilization for elastic tasks."""
       if self.utilization_elastic and self.utilization_elastic[core_index] is not None:
            min_u, max_u = self.utilization_elastic[core_index]
            if min_u is not None and max_u is not None:
                return (min_u + max_u) / 2.0
       elif self.utilization_fixed and self.utilization_fixed[core_index] is not None:
           return self.utilization_fixed[core_index]
       return None

    def get_min_utilization(self, core_index: int) -> Optional[float]:
        if self.utilization_elastic and self.utilization_elastic[core_index] is not None:
            return self.utilization_elastic[core_index][0]
        elif self.utilization_fixed and self.utilization_fixed[core_index] is not None:
            return self.utilization_fixed[core_index] # Fixed tasks have min=max=fixed
        return None

    def get_max_utilization(self, core_index: int) -> Optional[float]:
        if self.utilization_elastic and self.utilization_elastic[core_index] is not None:
            return self.utilization_elastic[core_index][1]
        elif self.utilization_fixed and self.utilization_fixed[core_index] is not None:
            return self.utilization_fixed[core_index] # Fixed tasks have min=max=fixed
        return None


@dataclass
class Core:
    id: int
    assigned_cluster_id: Optional[int] = None
    # We can add speed factor later if needed, but heterogeneity
    # is currently modeled by task utilization differences.

@dataclass
class Cluster:
    id: int
    core_ids: List[int]
    tasks: List[Task] = field(default_factory=list)
    task_assignments: Dict[int, List[int]] = field(default_factory=dict) # {task_id: [core_id1, core_id2/None]}
    # Capacity is implicitly 1.0 per core, so 2.0 for a 2-core cluster.
    # We'll check utilization sums against this.

# --- Task Set Generation ---

def generate_task_set(num_tasks, num_cores, target_uf, util_mean=0.4, util_std=0.2, period_range=(10, 100), elastic_factor=0.2):
    tasks_cost = []
    tasks_elastic = []
    periods = []

    # Generate base execution times (needed to derive utilizations later)
    # Use Normal dist (mean 50, std 10, but this seems high for utils)
    # Let's generate nominal utilizations directly based on paper's util params
    # And generate periods
    
    for i in range(num_tasks):
         period = random.randint(period_range[0], period_range[1])
         # Ensure periods are multiples of 10 for simpler hyperperiods? Or random? Let's stick to random for now.
         periods.append(period)
         
         task_id = i
         
         # Generate nominal utilizations per core
         nominal_utils = np.random.normal(util_mean, util_std, num_cores)
         nominal_utils = np.clip(nominal_utils, 0.05, 0.95) # Ensure reasonable bounds

         util_fixed = [float(u) for u in nominal_utils]
         util_elastic = []
         for u_nom in nominal_utils:
             u_min = max(0.01, u_nom * (1.0 - elastic_factor)) # Min 80% of nominal
             u_max = min(1.0, u_nom * (1.0 + elastic_factor)) # Max 120% of nominal
             if u_min > u_max: # Handle edge case where factor makes min > max
                 u_min = u_max = u_nom
             util_elastic.append( (float(u_min), float(u_max)) )

         tasks_cost.append(Task(id=task_id, period=period, utilization_fixed=util_fixed))
         tasks_elastic.append(Task(id=task_id, period=period, utilization_elastic=util_elastic))

    # --- Scale utilizations to meet target UF ---
    # Calculate current UF based on nominal utilizations
    current_uf_cost = calculate_uf(tasks_cost, num_cores)
    current_uf_elastic = calculate_uf(tasks_elastic, num_cores)
    # Use average UF for scaling factor determination
    current_uf = (current_uf_cost + current_uf_elastic) / 2.0

    if current_uf > 0:
        scaling_factor = target_uf / current_uf
        # Apply scaling and re-clip
        for i in range(num_tasks):
            # Scale COST tasks
            scaled_fixed = [min(1.0, max(0.01, u * scaling_factor)) if u is not None else None for u in tasks_cost[i].utilization_fixed]
            tasks_cost[i].utilization_fixed = scaled_fixed
            
            # Scale COST-Elastic tasks (scale both min and max bounds)
            scaled_elastic = []
            for u_min, u_max in tasks_elastic[i].utilization_elastic:
                 if u_min is not None and u_max is not None:
                     scaled_u_min = min(1.0, max(0.01, u_min * scaling_factor))
                     scaled_u_max = min(1.0, max(0.01, u_max * scaling_factor))
                     if scaled_u_min > scaled_u_max: # Adjust if scaling inverted bounds
                         scaled_u_min = scaled_u_max
                     scaled_elastic.append((scaled_u_min, scaled_u_max))
                 else:
                     scaled_elastic.append((None,None)) # Task cannot run on this core
            tasks_elastic[i].utilization_elastic = scaled_elastic
            
            # Update periods (they should be the same)
            tasks_cost[i].period = tasks_elastic[i].period = periods[i]

    # Verify scaled UF (optional)
    # final_uf_cost = calculate_uf(tasks_cost, num_cores)
    # final_uf_elastic = calculate_uf(tasks_elastic, num_cores)
    # print(f"Target UF: {target_uf}, Initial UF: {current_uf:.2f}, Final UF COST: {final_uf_cost:.2f}, Final UF Elastic: {final_uf_elastic:.2f}")

    cores = [Core(id=j) for j in range(num_cores)]
    
    return tasks_cost, tasks_elastic, cores, periods


# --- COST Algorithm Implementation ---

class COSTScheduler:
    def __init__(self, tasks: List[Task], cores: List[Core]):
        self.tasks = copy.deepcopy(tasks)
        self.cores = copy.deepcopy(cores)
        self.num_tasks = len(tasks)
        self.num_cores = len(cores)
        self.clusters: Dict[int, Cluster] = {}
        self.task_cluster_map: Dict[int, int] = {} # task_id -> cluster_id
        self.core_cluster_map: Dict[int, int] = {} # core_id -> cluster_id

    def form_clusters(self) -> bool:
        """Implements FORM-CLUSTERS logic from COST paper."""
        unassigned_tasks = {task.id for task in self.tasks}
        assigned_cores = set()
        cluster_id_counter = 0

        # Store (util, task_id, core_id) tuples
        util_list = []
        for task in self.tasks:
            for core_idx, util in enumerate(task.utilization_fixed):
                 if util is not None:
                     util_list.append((util, task.id, core_idx))
        
        # Sort by utilization (ascending)
        util_list.sort()

        tasks_in_clusters = set()
        
        processed_tasks_in_sort = set() # Track tasks processed via util_list

        while len(tasks_in_clusters) < self.num_tasks :
            
            # Find the next task to assign based on sorted util_list
            # ensuring the task hasn't been assigned yet
            next_util_info = None
            idx_to_process = -1
            for idx, (util, task_id, core_id) in enumerate(util_list):
                 if task_id not in tasks_in_clusters and task_id not in processed_tasks_in_sort:
                       next_util_info = (util, task_id, core_id)
                       idx_to_process = idx
                       processed_tasks_in_sort.add(task_id)
                       break # Found the lowest util unassigned task
            
            if next_util_info is None:
                 # This might happen if some tasks cannot be placed anywhere or all tasks are processed.
                 # Or if a task failed assignment earlier.
                 break # No more assignable tasks found this way

            util, task_id, core1_id = next_util_info
            current_task = next(t for t in self.tasks if t.id == task_id)

            # Case 1: Core1 is already in a cluster
            if core1_id in self.core_cluster_map:
                target_cluster_id = self.core_cluster_map[core1_id]
                target_cluster = self.clusters[target_cluster_id]
                
                # Check capacity (Simplified: sum of avg utils <= num_cores_in_cluster)
                # COST paper uses Hetero-Split/Wrap which is complex.
                # We simplify: Check if adding this task's *average* utilization across
                # the cluster's cores is feasible.
                
                cluster_core_ids = target_cluster.core_ids
                avg_task_util_in_cluster = np.mean([current_task.utilization_fixed[c_id] 
                                                    for c_id in cluster_core_ids if current_task.utilization_fixed[c_id] is not None])
                
                current_cluster_util = 0.0
                for t in target_cluster.tasks:
                     current_cluster_util += np.mean([t.utilization_fixed[c_id] 
                                                     for c_id in cluster_core_ids if t.utilization_fixed[c_id] is not None])

                # Capacity is roughly len(cluster_core_ids) * 1.0
                if current_cluster_util + avg_task_util_in_cluster <= len(cluster_core_ids):
                    target_cluster.tasks.append(current_task)
                    self.task_cluster_map[task_id] = target_cluster_id
                    current_task.assigned_cluster_id = target_cluster_id
                    tasks_in_clusters.add(task_id)
                    # print(f"COST: Task {task_id} assigned to existing Cluster {target_cluster_id}")

                else:
                    # print(f"COST: Task {task_id} could not fit in existing Cluster {target_cluster_id} (capacity check)")
                    # Task cannot be assigned for now, may fail scheduling.
                    # Let loop continue to find other tasks/clusters. If it remains unassigned, it fails.
                     pass # Mark as processed but not assigned


            # Case 2: Core1 is not in any cluster yet
            elif core1_id not in assigned_cores:
                # Find the *next* best core (core2) for this task
                core2_id = -1
                min_util_core2 = float('inf')
                
                for c_idx, u in enumerate(current_task.utilization_fixed):
                     if u is not None and c_idx != core1_id and c_idx not in assigned_cores:
                         if u < min_util_core2:
                             min_util_core2 = u
                             core2_id = c_idx
                             
                # Form a new cluster
                new_cluster_id = cluster_id_counter
                cluster_cores = [core1_id]
                assigned_cores.add(core1_id)
                self.core_cluster_map[core1_id] = new_cluster_id
                self.cores[core1_id].assigned_cluster_id = new_cluster_id

                if core2_id != -1:
                    cluster_cores.append(core2_id)
                    assigned_cores.add(core2_id)
                    self.core_cluster_map[core2_id] = new_cluster_id
                    self.cores[core2_id].assigned_cluster_id = new_cluster_id
                # else: Cluster with only one core if no other available core found

                new_cluster = Cluster(id=new_cluster_id, core_ids=cluster_cores, tasks=[current_task])
                self.clusters[new_cluster_id] = new_cluster
                self.task_cluster_map[task_id] = new_cluster_id
                current_task.assigned_cluster_id = new_cluster_id
                tasks_in_clusters.add(task_id)
                cluster_id_counter += 1
                # print(f"COST: Task {task_id} formed new Cluster {new_cluster_id} with cores {cluster_cores}")
            
            # else: Core1 was considered but is now assigned (race condition?), ignore and continue loop.
            else:
                 pass # Core already assigned by another task's logic concurrently. Skip.


        # Check if all tasks were assigned
        return len(tasks_in_clusters) == self.num_tasks

    def schedule(self) -> bool:
        """Simplified scheduling check for COST."""
        if not self.form_clusters():
             # print("COST: Clustering failed, not all tasks assigned.")
             return False # Clustering failed

        # Check schedulability per cluster (Simplified)
        # A full implementation requires Hetero-Split/Wrap and frame-by-frame scheduling.
        # Simplification: Check if the sum of utilizations in each cluster exceeds its capacity.
        for cluster_id, cluster in self.clusters.items():
            total_util_cluster = 0.0
            num_cluster_cores = len(cluster.core_ids)
            
            # Check capacity using average utilization on cluster cores
            for task in cluster.tasks:
                 utils_on_cluster_cores = [task.utilization_fixed[c_id] for c_id in cluster.core_ids if task.utilization_fixed[c_id] is not None]
                 if not utils_on_cluster_cores:
                      # print(f"COST: Warning - Task {task.id} in Cluster {cluster_id} cannot run on any core in the cluster?")
                      return False # Should not happen if clustering worked correctly
                 avg_util = np.mean(utils_on_cluster_cores)
                 total_util_cluster += avg_util
                 
            # Rough check: Total average utilization <= number of cores
            if total_util_cluster > num_cluster_cores + 1e-9: # Add tolerance for float issues
                # print(f"COST: Cluster {cluster_id} failed schedulability check (Total Avg Util {total_util_cluster:.2f} > Capacity {num_cluster_cores})")
                return False

        # print("COST: Task set deemed schedulable (simplified check).")
        return True

# --- COST-Elastic Algorithm Implementation ---

class COSTElasticScheduler:
    def __init__(self, tasks: List[Task], cores: List[Core]):
        self.tasks = copy.deepcopy(tasks)
        self.cores = copy.deepcopy(cores)
        self.num_tasks = len(tasks)
        self.num_cores = len(cores)
        self.clusters: Dict[int, Cluster] = {}
        self.task_cluster_map: Dict[int, int] = {} # task_id -> cluster_id
        self.core_cluster_map: Dict[int, int] = {} # core_id -> cluster_id
        self.total_utilization_adjustment = 0.0
        self.num_adjustments = 0

    def form_clusters_elastic(self) -> bool:
        
        assigned_cores = set()
        cluster_id_counter = 0

        # Calculate elasticity potential for all tasks
        task_elasticity = {task.id: task.get_elasticity_potential(self.num_cores) for task in self.tasks}
        
        # Sort tasks by elasticity potential (descending)
        sorted_tasks_by_elasticity = sorted(self.tasks, key=lambda t: task_elasticity[t.id], reverse=True)

        tasks_in_clusters = set()

        for task in sorted_tasks_by_elasticity:
            if task.id in tasks_in_clusters:
                continue # Already assigned

            task_id = task.id

            # Find core with lowest *minimum* utilization for this task among unclustered cores
            best_core_id = -1
            lowest_min_util = float('inf')

            for core_idx in range(self.num_cores):
                if core_idx not in assigned_cores:
                     min_util = task.get_min_utilization(core_idx)
                     if min_util is not None and min_util < lowest_min_util:
                         lowest_min_util = min_util
                         best_core_id = core_idx

            if best_core_id == -1:
                 # Maybe task cannot run on any *available* core?
                 # Try finding the best core among *all* cores (even clustered ones)
                 # to check if it *could* fit in an existing cluster
                 temp_best_core_id = -1
                 temp_lowest_min_util = float('inf')
                 for core_idx in range(self.num_cores):
                     min_util = task.get_min_utilization(core_idx)
                     if min_util is not None and min_util < temp_lowest_min_util:
                         temp_lowest_min_util = min_util
                         temp_best_core_id = core_idx
                 
                 if temp_best_core_id != -1 and temp_best_core_id in self.core_cluster_map:
                      # Check if it fits in the existing cluster of the best core overall
                      target_cluster_id = self.core_cluster_map[temp_best_core_id]
                      target_cluster = self.clusters[target_cluster_id]
                      
                      # Capacity check (Adaptive Partitioning - Simplified)
                      # Can we assign *at least minimum* utilizations such that total <= capacity?
                      can_fit = self.check_cluster_capacity_elastic(target_cluster, task_to_add=task)
                      
                      if can_fit:
                          target_cluster.tasks.append(task)
                          self.task_cluster_map[task_id] = target_cluster_id
                          task.assigned_cluster_id = target_cluster_id
                          tasks_in_clusters.add(task_id)
                          # print(f"COST-E: Task {task_id} (high elasticity) assigned to existing Cluster {target_cluster_id}")
                          continue # Move to next task in elasticity sort
                      else:
                           # print(f"COST-E: Task {task_id} could not fit in existing Cluster {target_cluster_id} even with min util.")
                           pass # Fails clustering for this task

                 # If still no suitable core/cluster found, task fails
                 # print(f"COST-E: Task {task_id} cannot be placed on any available core/cluster.")
                 continue # Cannot place this task

            # --- Found an unclustered core (best_core_id) ---
            core1_id = best_core_id

            # Find the next best *unclustered* core (core2) for this task based on min_util
            core2_id = -1
            lowest_min_util_core2 = float('inf')
            for c_idx in range(self.num_cores):
                 if c_idx != core1_id and c_idx not in assigned_cores:
                      min_util = task.get_min_utilization(c_idx)
                      if min_util is not None and min_util < lowest_min_util_core2:
                          lowest_min_util_core2 = min_util
                          core2_id = c_idx
            
            # Form a new cluster
            new_cluster_id = cluster_id_counter
            cluster_cores = [core1_id]
            assigned_cores.add(core1_id)
            self.core_cluster_map[core1_id] = new_cluster_id
            self.cores[core1_id].assigned_cluster_id = new_cluster_id

            if core2_id != -1:
                 cluster_cores.append(core2_id)
                 assigned_cores.add(core2_id)
                 self.core_cluster_map[core2_id] = new_cluster_id
                 self.cores[core2_id].assigned_cluster_id = new_cluster_id
            
            new_cluster = Cluster(id=new_cluster_id, core_ids=cluster_cores, tasks=[task])
            self.clusters[new_cluster_id] = new_cluster
            self.task_cluster_map[task_id] = new_cluster_id
            task.assigned_cluster_id = new_cluster_id
            tasks_in_clusters.add(task_id)
            cluster_id_counter += 1
            # print(f"COST-E: Task {task_id} (high elasticity) formed new Cluster {new_cluster_id} with cores {cluster_cores}")


        # Assign remaining tasks (lower elasticity) using COST-like logic (lowest min_util)
        remaining_tasks = [t for t in self.tasks if t.id not in tasks_in_clusters]
        
        util_list_remaining = []
        for task in remaining_tasks:
             for core_idx in range(self.num_cores):
                  min_util = task.get_min_utilization(core_idx)
                  if min_util is not None:
                      util_list_remaining.append((min_util, task.id, core_idx))
        util_list_remaining.sort()
        
        processed_remaining_tasks = set()

        while len(tasks_in_clusters) < self.num_tasks and util_list_remaining:
             
             next_util_info = None
             idx_to_process = -1
             
             found_unprocessed = False
             for idx, (util, task_id, core_id) in enumerate(util_list_remaining):
                  if task_id not in tasks_in_clusters and task_id not in processed_remaining_tasks:
                       next_util_info = (util, task_id, core_id)
                       processed_remaining_tasks.add(task_id) # Mark as processed
                       found_unprocessed = True
                       break 
            
             if not found_unprocessed:
                  break # All remaining tasks processed or unassignable

             min_util, task_id, core1_id = next_util_info
             current_task = next(t for t in self.tasks if t.id == task_id)

             # Case 1: Core1 is already in a cluster
             if core1_id in self.core_cluster_map:
                 target_cluster_id = self.core_cluster_map[core1_id]
                 target_cluster = self.clusters[target_cluster_id]
                 
                 can_fit = self.check_cluster_capacity_elastic(target_cluster, task_to_add=current_task)
                 
                 if can_fit:
                     target_cluster.tasks.append(current_task)
                     self.task_cluster_map[task_id] = target_cluster_id
                     current_task.assigned_cluster_id = target_cluster_id
                     tasks_in_clusters.add(task_id)
                     # print(f"COST-E: Task {task_id} (lower elasticity) assigned to existing Cluster {target_cluster_id}")
                 else:
                     # print(f"COST-E: Task {task_id} (lower elasticity) could not fit in existing Cluster {target_cluster_id}.")
                      pass # Cannot fit

             # Case 2: Core1 is not in any cluster yet (Should not happen often if previous phase worked)
             elif core1_id not in assigned_cores:
                 # Find the *next* best unclustered core (core2)
                 core2_id = -1
                 lowest_min_util_core2 = float('inf')
                 for c_idx in range(self.num_cores):
                      if c_idx != core1_id and c_idx not in assigned_cores:
                           min_util_c2 = current_task.get_min_utilization(c_idx)
                           if min_util_c2 is not None and min_util_c2 < lowest_min_util_core2:
                               lowest_min_util_core2 = min_util_c2
                               core2_id = c_idx
                 
                 new_cluster_id = cluster_id_counter
                 cluster_cores = [core1_id]
                 assigned_cores.add(core1_id)
                 self.core_cluster_map[core1_id] = new_cluster_id
                 self.cores[core1_id].assigned_cluster_id = new_cluster_id

                 if core2_id != -1:
                     cluster_cores.append(core2_id)
                     assigned_cores.add(core2_id)
                     self.core_cluster_map[core2_id] = new_cluster_id
                     self.cores[core2_id].assigned_cluster_id = new_cluster_id
                 
                 new_cluster = Cluster(id=new_cluster_id, core_ids=cluster_cores, tasks=[current_task])
                 self.clusters[new_cluster_id] = new_cluster
                 self.task_cluster_map[task_id] = new_cluster_id
                 current_task.assigned_cluster_id = new_cluster_id
                 tasks_in_clusters.add(task_id)
                 cluster_id_counter += 1
                 # print(f"COST-E: Task {task_id} (lower elasticity) formed new Cluster {new_cluster_id} with cores {cluster_cores}")

             else:
                  pass # Core already assigned, skip.


        return len(tasks_in_clusters) == self.num_tasks

    def check_cluster_capacity_elastic(self, cluster: Cluster, task_to_add: Optional[Task] = None) -> bool:
        """Checks if tasks (including optional new one) can fit using minimum utilizations."""
        total_min_util = 0.0
        tasks_to_check = cluster.tasks + ([task_to_add] if task_to_add else [])
        
        if not tasks_to_check:
             return True # Empty cluster is fine

        # Simplified check: Sum of minimum utilizations <= cluster capacity
        # A more accurate check involves solving a small allocation problem.
        
        for task in tasks_to_check:
             min_utils_on_cluster = [task.get_min_utilization(c_id) for c_id in cluster.core_ids if task.get_min_utilization(c_id) is not None]
             if not min_utils_on_cluster:
                 # print(f"COST-E Check: Task {task.id} cannot run on any core in cluster {cluster.id}")
                 return False # Task cannot run here
             # Use the average minimum utilization as a proxy for its load
             avg_min_util = np.mean(min_utils_on_cluster)
             total_min_util += avg_min_util

        return total_min_util <= len(cluster.core_ids) + 1e-9


    def adaptive_partition_and_schedule(self) -> Tuple[bool, float, float]:
        """Assigns utilizations within elastic bounds and checks schedulability."""
        if not self.form_clusters_elastic():
            # print("COST-E: Elastic clustering failed.")
            return False, 0.0, 0.0 # Clustering failed

        total_cluster_utilization = 0.0
        num_clusters_measured = 0
        
        schedulable = True
        for cluster_id, cluster in self.clusters.items():
            cluster_cores = cluster.core_ids
            num_cluster_cores = len(cluster_cores)
            cluster_capacity = float(num_cluster_cores)
            
            # Try to assign utilizations (Adaptive Partitioning - Simplified)
            # Start with minimum utilization for all tasks in the cluster
            current_assignment: Dict[int, float] = {} # task_id -> assigned_util
            current_cluster_load = 0.0
            
            tasks_in_cluster = sorted(cluster.tasks, key=lambda t: t.get_elasticity_potential(self.num_cores), reverse=True) # Prioritize elastic tasks for increase
            
            possible_assignment = True
            for task in tasks_in_cluster:
                 min_utils_on_cluster = [task.get_min_utilization(c_id) for c_id in cluster_cores if task.get_min_utilization(c_id) is not None]
                 if not min_utils_on_cluster:
                      possible_assignment = False
                      break
                 # Assign the average minimum utilization initially
                 assigned_util = np.mean(min_utils_on_cluster)
                 current_assignment[task.id] = assigned_util
                 current_cluster_load += assigned_util

            if not possible_assignment or current_cluster_load > cluster_capacity + 1e-9:
                 schedulable = False
                 # print(f"COST-E: Cluster {cluster_id} unschedulable even with minimum utilizations (Load {current_cluster_load:.2f} > Cap {cluster_capacity:.1f})")
                 break # This cluster fails

            # Elasticity-Aware Scheduling (Simplified): Distribute remaining capacity
            remaining_capacity = cluster_capacity - current_cluster_load
            if remaining_capacity > 1e-9:
                
                total_elasticity_in_cluster = 0.0
                task_elasticity_remaining: Dict[int, float] = {}

                for task in tasks_in_cluster:
                     avg_max_util = np.mean([task.get_max_utilization(c_id) for c_id in cluster_cores if task.get_max_utilization(c_id) is not None])
                     if avg_max_util is not None :
                           elastic_range = avg_max_util - current_assignment[task.id]
                           if elastic_range > 0:
                                task_elasticity_remaining[task.id] = elastic_range
                                total_elasticity_in_cluster += elastic_range

                if total_elasticity_in_cluster > 1e-9:
                    # Distribute remaining capacity proportionally to remaining elasticity
                    for task in tasks_in_cluster:
                        if task.id in task_elasticity_remaining:
                            share = task_elasticity_remaining[task.id] / total_elasticity_in_cluster
                            increase = share * remaining_capacity
                            
                            avg_max_util = np.mean([task.get_max_utilization(c_id) for c_id in cluster_cores if task.get_max_utilization(c_id) is not None])
                            
                            # Ensure we don't exceed max bound
                            new_util = min(avg_max_util, current_assignment[task.id] + increase)
                            actual_increase = new_util - current_assignment[task.id]
                            
                            current_assignment[task.id] = new_util
                            current_cluster_load += actual_increase # Update load based on actual increase

            # Final check after elastic adjustment
            if current_cluster_load > cluster_capacity + 1e-9:
                 schedulable = False
                 # print(f"COST-E: Cluster {cluster_id} unschedulable after elastic adjustments (Load {current_cluster_load:.2f} > Cap {cluster_capacity:.1f})")
                 break
            else:
                 # Record final assignments and calculate adjustments
                 cluster_util_sum = 0.0
                 for task in cluster.tasks:
                      task.current_utilization = [None] * self.num_cores # Initialize
                      assigned_u = current_assignment.get(task.id, 0.0) # Should always be found
                      
                      # Distribute the average assigned util back to cores (approximation)
                      # A better way would track per-core load during assignment
                      num_valid_cores = sum(1 for c_id in cluster_cores if task.get_min_utilization(c_id) is not None)
                      util_per_core = assigned_u / num_valid_cores if num_valid_cores > 0 else 0
                      
                      for c_id in cluster_cores:
                           if task.get_min_utilization(c_id) is not None:
                              task.current_utilization[c_id] = util_per_core

                      nominal_util = np.mean([task.get_nominal_utilization(c_id) for c_id in cluster_cores if task.get_nominal_utilization(c_id) is not None])
                      if nominal_util is not None:
                          self.total_utilization_adjustment += abs(assigned_u - nominal_util)
                          self.num_adjustments += 1
                      cluster_util_sum += assigned_u # Sum of average assigned utils
                 
                 avg_cluster_util = cluster_util_sum / num_cluster_cores if num_cluster_cores > 0 else 0
                 total_cluster_utilization += avg_cluster_util
                 num_clusters_measured += 1
                 # print(f"COST-E: Cluster {cluster_id} schedulable (Final Load {current_cluster_load:.2f})")


        avg_cluster_util_overall = total_cluster_utilization / num_clusters_measured if num_clusters_measured > 0 else 0.0
        avg_util_adjustment = self.total_utilization_adjustment / self.num_adjustments if self.num_adjustments > 0 else 0.0

        if not schedulable:
             return False, 0.0, 0.0 # Return 0 if scheduling failed

        return schedulable, avg_cluster_util_overall, avg_util_adjustment


# --- Simulation ---

def run_simulation(num_trials, num_tasks_list, num_cores_list, uf_list):
    results = {
        'COST': {'acceptance_ratio': {}, 'cluster_util': {}},
        'COST-Elastic': {'acceptance_ratio': {}, 'cluster_util': {}, 'avg_adjustment': {}}
    }
    param_keys = [] # To store (param_type, param_value) for indexing results

    # Experiment 1: Varying Utilization Factor
    print("\n--- Experiment 1: Varying Utilization Factor ---")
    exp1_cores = 4
    exp1_tasks = 20
    param_type = "UF"
    results['COST']['acceptance_ratio'][param_type] = []
    results['COST']['cluster_util'][param_type] = []
    results['COST-Elastic']['acceptance_ratio'][param_type] = []
    results['COST-Elastic']['cluster_util'][param_type] = []
    results['COST-Elastic']['avg_adjustment'][param_type] = []
    param_keys.append(param_type)

    uf_values_exp1 = sorted(uf_list)

    for uf in uf_values_exp1:
        cost_accepted = 0
        elastic_accepted = 0
        cost_cluster_utils = []
        elastic_cluster_utils = []
        elastic_adjustments = []

        print(f"  Running UF = {uf:.2f}...")
        for trial in range(num_trials):
            tasks_cost, tasks_elastic, cores, periods = generate_task_set(
                exp1_tasks, exp1_cores, uf
            )

            # Run COST
            cost_scheduler = COSTScheduler(tasks_cost, cores)
            if cost_scheduler.schedule():
                cost_accepted += 1
                # Calculate COST cluster utilization (simplified: average core util)
                total_util_cost = sum(np.mean([u for u in t.utilization_fixed if u is not None]) for t in cost_scheduler.tasks if t.utilization_fixed)
                cost_cluster_utils.append(total_util_cost / exp1_cores if exp1_cores > 0 else 0)


            # Run COST-Elastic
            elastic_scheduler = COSTElasticScheduler(tasks_elastic, cores)
            schedulable_e, avg_clust_util_e, avg_adj_e = elastic_scheduler.adaptive_partition_and_schedule()
            if schedulable_e:
                elastic_accepted += 1
                elastic_cluster_utils.append(avg_clust_util_e)
                elastic_adjustments.append(avg_adj_e)
            # else: Keep utils/adjustments as 0 if it failed


        results['COST']['acceptance_ratio'][param_type].append(cost_accepted / num_trials)
        results['COST']['cluster_util'][param_type].append(np.mean(cost_cluster_utils) if cost_cluster_utils else 0)
        results['COST-Elastic']['acceptance_ratio'][param_type].append(elastic_accepted / num_trials)
        results['COST-Elastic']['cluster_util'][param_type].append(np.mean(elastic_cluster_utils) if elastic_cluster_utils else 0)
        results['COST-Elastic']['avg_adjustment'][param_type].append(np.mean(elastic_adjustments) if elastic_adjustments else 0)

    # Experiment 2: Varying Number of Cores (Using fixed UF from COST-Elastic paper, e.g., 0.7)
    print("\n--- Experiment 2: Varying Number of Cores ---")
    exp2_uf = 0.7
    exp2_tasks = 20
    param_type = "Cores"
    results['COST']['acceptance_ratio'][param_type] = []
    results['COST']['cluster_util'][param_type] = []
    results['COST-Elastic']['acceptance_ratio'][param_type] = []
    results['COST-Elastic']['cluster_util'][param_type] = []
    results['COST-Elastic']['avg_adjustment'][param_type] = []
    param_keys.append(param_type)

    core_values_exp2 = sorted(num_cores_list)

    for n_cores in core_values_exp2:
        cost_accepted = 0
        elastic_accepted = 0
        cost_cluster_utils = []
        elastic_cluster_utils = []
        elastic_adjustments = []

        print(f"  Running Cores = {n_cores}...")
        for trial in range(num_trials):
            tasks_cost, tasks_elastic, cores, periods = generate_task_set(
                exp2_tasks, n_cores, exp2_uf
            )

            cost_scheduler = COSTScheduler(tasks_cost, cores)
            if cost_scheduler.schedule():
                cost_accepted += 1
                total_util_cost = sum(np.mean([u for u in t.utilization_fixed if u is not None]) for t in cost_scheduler.tasks if t.utilization_fixed)
                cost_cluster_utils.append(total_util_cost / n_cores if n_cores > 0 else 0)

            elastic_scheduler = COSTElasticScheduler(tasks_elastic, cores)
            schedulable_e, avg_clust_util_e, avg_adj_e = elastic_scheduler.adaptive_partition_and_schedule()
            if schedulable_e:
                elastic_accepted += 1
                elastic_cluster_utils.append(avg_clust_util_e)
                elastic_adjustments.append(avg_adj_e)

        results['COST']['acceptance_ratio'][param_type].append(cost_accepted / num_trials)
        results['COST']['cluster_util'][param_type].append(np.mean(cost_cluster_utils) if cost_cluster_utils else 0)
        results['COST-Elastic']['acceptance_ratio'][param_type].append(elastic_accepted / num_trials)
        results['COST-Elastic']['cluster_util'][param_type].append(np.mean(elastic_cluster_utils) if elastic_cluster_utils else 0)
        results['COST-Elastic']['avg_adjustment'][param_type].append(np.mean(elastic_adjustments) if elastic_adjustments else 0)


    # Experiment 3: Varying Number of Tasks (Using fixed UF and Cores)
    print("\n--- Experiment 3: Varying Number of Tasks ---")
    exp3_uf = 0.7
    exp3_cores = 4
    param_type = "Tasks"
    results['COST']['acceptance_ratio'][param_type] = []
    results['COST']['cluster_util'][param_type] = []
    results['COST-Elastic']['acceptance_ratio'][param_type] = []
    results['COST-Elastic']['cluster_util'][param_type] = []
    results['COST-Elastic']['avg_adjustment'][param_type] = []
    param_keys.append(param_type)

    task_values_exp3 = sorted(num_tasks_list)

    for n_tasks in task_values_exp3:
        cost_accepted = 0
        elastic_accepted = 0
        cost_cluster_utils = []
        elastic_cluster_utils = []
        elastic_adjustments = []

        print(f"  Running Tasks = {n_tasks}...")
        for trial in range(num_trials):
            tasks_cost, tasks_elastic, cores, periods = generate_task_set(
                n_tasks, exp3_cores, exp3_uf
            )

            cost_scheduler = COSTScheduler(tasks_cost, cores)
            if cost_scheduler.schedule():
                cost_accepted += 1
                total_util_cost = sum(np.mean([u for u in t.utilization_fixed if u is not None]) for t in cost_scheduler.tasks if t.utilization_fixed)
                cost_cluster_utils.append(total_util_cost / exp3_cores if exp3_cores > 0 else 0)


            elastic_scheduler = COSTElasticScheduler(tasks_elastic, cores)
            schedulable_e, avg_clust_util_e, avg_adj_e = elastic_scheduler.adaptive_partition_and_schedule()
            if schedulable_e:
                elastic_accepted += 1
                elastic_cluster_utils.append(avg_clust_util_e)
                elastic_adjustments.append(avg_adj_e)

        results['COST']['acceptance_ratio'][param_type].append(cost_accepted / num_trials)
        results['COST']['cluster_util'][param_type].append(np.mean(cost_cluster_utils) if cost_cluster_utils else 0)
        results['COST-Elastic']['acceptance_ratio'][param_type].append(elastic_accepted / num_trials)
        results['COST-Elastic']['cluster_util'][param_type].append(np.mean(elastic_cluster_utils) if elastic_cluster_utils else 0)
        results['COST-Elastic']['avg_adjustment'][param_type].append(np.mean(elastic_adjustments) if elastic_adjustments else 0)


    return results, uf_values_exp1, core_values_exp2, task_values_exp3

# --- Plotting ---

def plot_results(results, uf_values, core_values, task_values):
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style

    # Plot 1: Acceptance Ratio vs. Utilization Factor
    param_type = "UF"
    plt.figure(figsize=(8, 6))
    plt.plot(uf_values, results['COST']['acceptance_ratio'][param_type], marker='o', linestyle='-', label='COST Acc. Ratio')
    plt.plot(uf_values, results['COST-Elastic']['acceptance_ratio'][param_type], marker='s', linestyle='--', label='COST-Elastic Acc. Ratio')
    plt.xlabel("Utilization Factor (UF)")
    plt.ylabel("Acceptance Ratio")
    plt.title("Acceptance Ratio vs. Utilization Factor (Cores=4, Tasks=20)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("acceptance_ratio_vs_uf.png")
    plt.show()

    # Plot 2: Cluster Utilization vs. Utilization Factor
    plt.figure(figsize=(8, 6))
    plt.plot(uf_values, results['COST']['cluster_util'][param_type], marker='o', linestyle='-', label='COST Avg. Cluster Util')
    plt.plot(uf_values, results['COST-Elastic']['cluster_util'][param_type], marker='s', linestyle='--', label='COST-Elastic Avg. Cluster Util')
    plt.xlabel("Utilization Factor (UF)")
    plt.ylabel("Average Cluster Utilization")
    plt.title("Cluster Utilization vs. Utilization Factor (Cores=4, Tasks=20)")
    plt.ylim(0, 1.0) # Utilization cannot exceed 1 per core on average
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cluster_util_vs_uf.png")
    plt.show()


    # Plot 3: Acceptance Ratio vs. Number of Cores
    param_type = "Cores"
    plt.figure(figsize=(8, 6))
    plt.plot(core_values, results['COST']['acceptance_ratio'][param_type], marker='o', linestyle='-', label='COST Acc. Ratio')
    plt.plot(core_values, results['COST-Elastic']['acceptance_ratio'][param_type], marker='s', linestyle='--', label='COST-Elastic Acc. Ratio')
    plt.xlabel("Number of Cores")
    plt.ylabel("Acceptance Ratio")
    plt.title(f"Acceptance Ratio vs. Number of Cores (UF=0.7, Tasks=20)")
    plt.ylim(0, 1.05)
    plt.xticks(core_values) # Ensure ticks match the data points
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("acceptance_ratio_vs_cores.png")
    plt.show()

    # Plot 4: Acceptance Ratio vs. Number of Tasks
    param_type = "Tasks"
    plt.figure(figsize=(8, 6))
    plt.plot(task_values, results['COST']['acceptance_ratio'][param_type], marker='o', linestyle='-', label='COST Acc. Ratio')
    plt.plot(task_values, results['COST-Elastic']['acceptance_ratio'][param_type], marker='s', linestyle='--', label='COST-Elastic Acc. Ratio')
    plt.xlabel("Number of Tasks")
    plt.ylabel("Acceptance Ratio")
    plt.title(f"Acceptance Ratio vs. Number of Tasks (UF=0.7, Cores=4)")
    plt.ylim(0, 1.05)
    plt.xticks(task_values)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("acceptance_ratio_vs_tasks.png")
    plt.show()

    # Plot 5: Average Utilization Adjustment (COST-Elastic only) vs UF
    param_type = "UF"
    plt.figure(figsize=(8, 6))
    plt.plot(uf_values, results['COST-Elastic']['avg_adjustment'][param_type], marker='s', linestyle='--', label='COST-Elastic Avg. Adjustment')
    plt.xlabel("Utilization Factor (UF)")
    plt.ylabel("Average Utilization Adjustment")
    plt.title("Avg. Utilization Adjustment vs. UF (COST-Elastic)")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("avg_adjustment_vs_uf.png")
    plt.show()
    
    # Plot 6: Makespan vs Number of Tasks (Conceptual - Not directly calculated here)
    # The papers don't provide a direct makespan calculation method suitable for this
    # simulation level. We infer makespan improvement from higher utilization/acceptance.
    # We will skip plotting makespan directly as the simulation doesn't compute it.
    print("\nNote: Makespan plot is omitted as the simulation focuses on schedulability checks,")
    print("      and direct makespan calculation is complex and not the primary metric ")
    print("      compared between the simplified schedulers implemented here.")


# --- Main Execution ---
if __name__ == "__main__":
    # Simulation Parameters (based on papers)
    NUM_TRIALS = 50 # Use 50 as in COST paper (100 in COST-Elastic) - 50 is faster
    
    # Parameter Ranges
    # UF_LIST = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # From COST Fig 5
    UF_LIST = np.linspace(0.5, 1.0, 6) # Matches COST-Elastic Fig 1 range better
    NUM_CORES_LIST = [2, 4, 6, 8] # From COST Fig 6
    NUM_TASKS_LIST = [10, 20, 30, 40] # From COST Fig 7

    # Task Generation Parameters
    UTIL_MEAN = 0.4 # From COST paper description
    UTIL_STD = 0.2  # From COST paper description
    # PERIOD_RANGE = (10, 100) # Reasonable range, not specified exactly
    # Let's use the example periods: 10, 20, 40 from COST paper's example to infer typical range
    PERIOD_RANGE = (10, 50) 
    ELASTIC_FACTOR = 0.2 # Corresponds to 80%-120% range in COST-Elastic paper

    print("Starting Simulation...")
    simulation_results, uf_vals, core_vals, task_vals = run_simulation(
        NUM_TRIALS, NUM_TASKS_LIST, NUM_CORES_LIST, UF_LIST
    )
    print("\nSimulation Complete.")

    print("\nPlotting Results...")
    plot_results(simulation_results, uf_vals, core_vals, task_vals)
    print("Plotting Complete. Check the saved PNG files.")

    # --- Brief Explanation ---
    print("\n--- Code Explanation ---")
    print("1.  **Inputs:** Task sets (periods, utilizations) generated randomly based on parameters")
    print("    described in the papers (Normal distributions for utilization, specified ranges")
    print("    for UF, cores, tasks). COST uses fixed utilizations, COST-Elastic uses ranges")
    print("    [nominal*(1-factor), nominal*(1+factor)]. Utilization Factor (UF) is targeted by scaling.")
    print("2.  **Algorithms:**")
    print("    - **COST:** Implements `form_clusters` based on lowest fixed utilization. Scheduling is")
    print("      simplified to a capacity check (sum of average utilizations <= cluster core count).")
    print("    - **COST-Elastic:** Implements `form_clusters_elastic` prioritizing high-elasticity tasks,")
    print("      then lowest minimum utilization. `adaptive_partition_and_schedule` attempts to assign")
    print("      utilizations within elastic bounds, starting from minimum and distributing slack")
    print("      proportionally to remaining elasticity. Schedulability checked against cluster capacity.")
    print("3.  **Simplifications:**")
    print("    - Hetero-Split/Wrap and detailed frame-by-frame scheduling are abstracted.")
    print("      Schedulability is based on overall utilization vs. capacity checks per cluster.")
    print("    - Task migration modeling is implicit in cluster assignment but not explicitly simulated.")
    print("    - Makespan is not directly calculated; improvement is inferred from acceptance ratio/utilization.")
    print("4.  **Outputs:** The simulation runs multiple trials for each parameter setting.")
    print("    - **Acceptance Ratio:** (Successful Trials / Total Trials) plotted against UF, Cores, Tasks.")
    print("    - **Cluster Utilization:** Average utilization across cores/clusters (for successful trials), plotted against UF.")
    print("    - **Average Utilization Adjustment:** Mean absolute difference between assigned and nominal")
    print("      utilization for COST-Elastic (for successful trials), plotted against UF.")
    print("    - Plots are generated using Matplotlib and saved as PNG files.")