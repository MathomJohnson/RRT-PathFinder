# Configuration Space Path Planner for UR5e Robot

## Project Context

This project implements a Rapidly-exploring Random Tree (RRT) path planner for a UR5e robotic arm in Webots simulation. The goal is to solve a pick-and-place task in a cluttered environment where the robot must move a block from a start location to a bin while navigating around static obstacles.

### Key Objectives
- **Task**: Pick and place a block in a cluttered environment with obstacles
- **Robot**: UR5e (6-DOF manipulator, 6-dimensional configuration space)
- **Algorithm**: RRT with goal biasing (10% probability)
- **Environment**: Webots simulation with static obstacles (walls/boxes)

### Critical Architectural Constraints

#### "Math Land" vs. "Simulation World"
The RRT planner operates entirely in "Math Land" (pure Python computation) without:
- Moving the simulated robot during search
- Querying the physics engine during planning
- Real-time Webots interaction during tree expansion

**Workflow**: Plan complete path in Python → Send commands to Webots for execution

#### Collision Detection Strategy (SIMPLIFIED)
- **Obstacles**: Defined as Axis-Aligned Bounding Boxes (AABB) in Python code
- **Robot Representation**: **End-effector position only** (not full arm)
- **Checking Method**: Test if end-effector 3D position lies inside obstacle AABBs
- **Simplification Rationale**: Robot arm joints can pass through obstacles; only the end-effector (and carried block) must avoid collisions. This focuses learning on RRT algorithm rather than complex collision geometry.
- **Webots Obstacles**: Visual "ghost" markers without physics collision

#### Coordinate Frame Management (CRITICAL)
This is one of the most important architectural decisions:

**The Problem:**
- **Webots world file (.wbt)**: Obstacles and objects are defined in WORLD coordinates (global frame)
- **Forward Kinematics (FK)**: Returns positions in ROBOT frame (local frame, origin at robot base)
- **Mismatch**: If you use world-frame obstacle coordinates with robot-frame FK positions, collision detection will be completely wrong!

**The Solution:**
1. **Find robot's world pose** from .wbt file:
   - Look for the UR5e node's `translation` field (e.g., `translation 0 2.2 0.7`)
   - Look for the UR5e node's `rotation` field (e.g., `rotation 0 0 1 1.570846325` = 90° around Z)

2. **Transform obstacles from world to robot frame**:
   - Read obstacle positions from .wbt file (world coordinates)
   - Subtract robot's translation: `point_relative = point_world - robot_base_world`
   - Apply inverse rotation: Rotate by `-robot_rotation_z` around Z-axis
   - Store transformed obstacles in `config.py` for collision detection

3. **All collision detection happens in robot frame**:
   - FK gives robot-frame positions
   - Obstacles are stored in robot frame
   - Both use the same coordinate system ✓

**Example:**
```python
# Robot pose from .wbt file
ROBOT_BASE_WORLD = np.array([0.0, 2.2, 0.7])
ROBOT_ROTATION_Z = 1.570846325  # ~90 degrees

# Transform function
def world_to_robot_frame(point_world):
    point_relative = point_world - ROBOT_BASE_WORLD
    angle = -ROBOT_ROTATION_Z
    x = point_relative[0] * cos(angle) - point_relative[1] * sin(angle)
    y = point_relative[0] * sin(angle) + point_relative[1] * cos(angle)
    return np.array([x, y, point_relative[2]])
```

**Where to implement:** All coordinate transformation logic goes in `config.py`, including both the transformation function and the transformed obstacle definitions.

#### Kinematics Requirements
- **Forward Kinematics (FK)**: Must return end-effector 3D position in ROBOT frame
- **Inverse Kinematics (IK)**: Used only to calculate target configuration q_goal for RRT goal biasing

## Code Style and Standards

### Python Best Practices
- **Type Hints**: Use comprehensive type annotations for all function signatures
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Array Convention**: Use NumPy arrays for joint angles, positions, and configurations
- **Validation**: Validate inputs at function boundaries, especially joint angle ranges

### Documentation Requirements
- Each function must document:
  - Purpose and algorithm overview
  - Parameters with types and units (radians for angles, meters for positions)
  - Return values with types and structure
  - Raises section for exceptions
  - Example usage where helpful

## Existing Code Resources

The project already contains valuable implementations from previous assignments:

Read the CLAUDE.md files inside hw2_code/, hw3_code/, and hw4_code/ folders to gain context on what is there.

### controllers/ur5e_controller/ - Current Active Controller
- **`kinematic_helpers.py`**: Active copy of DH parameters (import from here)
- **`ur5e_controller.py`**: Main Webots controller interface
- **`your_code_here.py`**: Where new RRT planning code should be implemented

### worlds/ - Webots Simulation Files
- **`hw2_ur5e.wbt`**: Existing UR5e world file (can be modified for this project)

## Project Architecture

### Current File Structure
```
RRT-PathFinder/
├── controllers/
│   └── ur5e_controller/
│       ├── kinematic_helpers.py     # UR5e DH parameters (ACTIVE)
│       ├── ur5e_controller.py       # Webots controller (ACTIVE)
│       └── your_code_here.py        # Main implementation file (ACTIVE)
├── hw2_code/                        # IK reference code
│   ├── kinematic_helpers.py         # DH parameters
│   ├── your_code_here.py            # Newton-Raphson IK
│   └── CLAUDE.md                    # IK documentation
├── hw3_code/                        # Trajectory + 2D RRT reference
│   ├── pygame_your_code_here.py     # 2D RRT implementation
│   ├── webots_your_code_here.py     # Cubic trajectory generation
│   ├── panda_controller.py
│   └── CLAUDE.md
├── hw4_code/                        # Vision-based grasping reference
│   ├── kinematic_helpers.py         # DH parameters
│   ├── your_code_here.py            # FSM + IK controller
│   └── CLAUDE.md
├── worlds/
│   └── hw2_ur5e.wbt                 # Webots world (modify for obstacles)
└── CLAUDE.md                        # This file
```

### New Files to Create During Project

You will create these new files as you progress through the sprints:

```
controllers/ur5e_controller/
├── collision.py              # AABB class and collision detection
├── rrt_planner.py            # Core RRT algorithm
├── path_utils.py             # Path smoothing and interpolation
└── config.py                 # Constants, parameters, obstacle definitions

Optional (for debugging/visualization):
├── visualizer.py             # Helper functions for Webots visualization
└── experiments.py            # Experiment runner for Sprint 6
```

### Module Design

#### `config.py` (Create in Sprint 1)
- **AABB class** - Simple data structure for axis-aligned bounding boxes
- Robot base position and rotation (from .wbt file)
- Coordinate frame transformation functions (world ↔ robot frame)
- Obstacle AABB definitions (in BOTH world and robot frames)
- UR5e joint limits
- RRT parameters (step_size, max_iterations, goal_bias, goal_threshold)
- Target pose/coordinates (bin location, start location)

#### `collision.py` (Sprint 1-2)
- Imports `AABB` class from `config.py`
- `is_point_in_aabb(point: np.ndarray, aabb: AABB) -> bool`: Basic point-in-box test
- `get_end_effector_position(q: np.ndarray) -> np.ndarray`: Returns end-effector 3D position in ROBOT frame
- `is_in_collision(q: np.ndarray, obstacles: List[AABB]) -> bool`: Checks if end-effector is inside any obstacle
- Uses FK from `kinematic_helpers.py` (T01, T12, T23, T34, T45, T56)

#### `rrt_planner.py` (Sprint 3-4)
- `class RRTNode`: Tree node with config and parent pointer
- `class RRTPlanner`:
  - `sample_configuration() -> np.ndarray`
  - `nearest_node(q_sample: np.ndarray) -> RRTNode`
  - `steer(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray`
  - `plan(q_start: np.ndarray, q_goal: np.ndarray) -> Optional[List[np.ndarray]]`
- Reference hw3_code/pygame_your_code_here.py for structure

#### `path_utils.py` (Sprint 5)
- `interpolate_path(path: List[np.ndarray], resolution: float) -> List[np.ndarray]`
- `smooth_path(path: List[np.ndarray], obstacles: List[AABB]) -> List[np.ndarray]`
- `validate_path(path: List[np.ndarray], obstacles: List[AABB]) -> bool`
- Can use cubic polynomials from hw3_code

#### `your_code_here.py` (Modified throughout)
- Main entry point called by `ur5e_controller.py`
- Orchestrates: planning → validation → execution
- Imports from collision.py, rrt_planner.py, path_utils.py

## Testing Strategy

### When to Use Math Verification (Minimal Unit Tests)
Use simple test scripts (not full pytest suites) only for:
1. **FK Correctness**: Print end-effector position for known configurations (home position), compare to UR5e specs
2. **Point-in-AABB**: Test 5-6 obvious cases (inside, outside, boundary) with print statements
3. **IK Round-Trip**: Test `IK(FK(q_test))` ≈ `q_test` for a few configurations
4. **Configuration Distance**: Verify distance metric is symmetric and satisfies triangle inequality

**Format**: Simple Python scripts that print PASS/FAIL, not formal test frameworks

### Primary Testing Method: Webots Visual Verification
For each sprint, the main validation is:
1. **Run the simulation** with test configurations
2. **Visually inspect** the robot behavior
3. **Use Supervisor node** to draw debug visualizations (end-effector positions, AABBs, paths)
4. **Add print statements** to log what's happening
5. **Success criteria**: Does it look right? Does the robot behave as expected?


## Sprint Plan

**IMPORTANT** - Work on one micro sprint at a time. DO NOT move on to the next micro sprint until permission is given by the user.

### Large Sprint 1: The Virtual World
**Goal**: Establish the mathematical model of the environment and verify it matches Webots visually.

#### Micro-Sprint 1.1: Define Obstacle Geometry and Coordinate Frames
- **Task**: Create `config.py` with AABB class and coordinate transformation
- **Steps**:
  1. Read obstacle position/size from .wbt file (world coordinates)
  2. Read robot's `translation` and `rotation` from .wbt file
  3. Implement `world_to_robot_frame()` transformation function
  4. Transform obstacle corners from world to robot frame
  5. Create AABB in robot frame from transformed corners
- **Deliverable**:
  ```python
  # config.py
  ROBOT_BASE_WORLD = np.array([0.0, 2.2, 0.7])  # From .wbt
  ROBOT_ROTATION_Z = 1.570846325  # From .wbt

  def world_to_robot_frame(point_world: np.ndarray) -> np.ndarray:
      # Transform point from world to robot frame
      pass

  # Obstacle in WORLD frame (from .wbt file)
  OBSTACLE_BOX_WORLD = AABB(min=[...], max=[...])

  # Transform to ROBOT frame for collision detection
  obstacle_corners_robot = [world_to_robot_frame(corner) for corner in corners]
  OBSTACLE_BOX_ROBOT = AABB(
      min=np.min(obstacle_corners_robot, axis=0),
      max=np.max(obstacle_corners_robot, axis=0)
  )

  # Use ROBOT frame obstacles for collision detection
  OBSTACLES = [OBSTACLE_BOX_ROBOT]
  ```
- **Validation**: Print both world-frame and robot-frame AABBs, verify transformation makes sense
- **Critical**: See "Coordinate Frame Management" section for why this is necessary!

#### Micro-Sprint 1.2: Create Basic Collision Module
- **Task**: Create `collision.py` with point-in-box collision detection
- **Note**: AABB class was already created in `config.py` in Sprint 1.1
- **Deliverable**: In `collision.py`:
  ```python
  from config import AABB
  import numpy as np

  def is_point_in_aabb(point: np.ndarray, aabb: AABB) -> bool:
      """Check if point is inside AABB."""
      return np.all(point >= aabb.min) and np.all(point <= aabb.max)
  ```
- **Test**: Import OBSTACLES from config.py, test with points inside/outside
- **Validation**: Print test results, verify boundary cases work correctly

#### Micro-Sprint 1.3: Implement End-Effector Forward Kinematics
- **Task**: Write function that returns end-effector 3D position in robot frame
- **Deliverable**: Add to `collision.py`:
  ```python
  def get_end_effector_position(q: np.ndarray) -> np.ndarray:
      """Returns end-effector position [x, y, z] in robot frame."""
      # Import T01, T12, etc. from kinematic_helpers
      # Chain transforms: T_base_to_ee = T01 @ T12 @ T23 @ T34 @ T45 @ T56
      # Extract position from final transform matrix
      pass
  ```
- **Test Case**: Call with q = [0, 0, 0, 0, 0, 0], print end-effector position
- **Validation**: Compare against expected UR5e workspace (typical reach ~0.85m from base)
- **Note**: This FK returns positions in ROBOT frame (origin at robot base)

#### Micro-Sprint 1.4: Webots Obstacle Visualization
- **Task**: Modify `ur5e_controller.py` to use Supervisor and draw obstacle AABB
- **Method**:
  - `from controller import Supervisor`
  - Use Supervisor field functions to add box primitive at obstacle location
  - OR use debug drawing functions if available
- **Deliverable**: When simulation runs, see a wireframe/solid box at obstacle location
- **Validation**: Obstacle appears where you expect in 3D space
- **Note**: Draw obstacle in WORLD frame coordinates (for visual reference)

#### Micro-Sprint 1.5: End-Effector Position Visualization (Optional)
- **Task**: Draw end-effector position (sphere) in Webots for debugging
- **Test Configurations**:
  - Home: [0, 0, 0, 0, 0, 0]
  - Extended: [0, -π/2, 0, 0, 0, 0]
  - Random safe config
- **Deliverable**: Visual sphere at computed end-effector position
- **Validation**: Sphere aligns with robot's actual gripper location in simulation
- **Debug**: If misaligned, check DH parameter usage and coordinate frames
- **Note**: This step is optional but helpful for verifying FK correctness

#### Micro-Sprint 1.6: Math-Visual Alignment Verification (Optional)
- **Task**: Command robot to random configurations, verify end-effector position stays aligned
- **Test**: Loop through 5-10 random joint angle sets, draw sphere at computed end-effector position
- **Success Criteria**: No visible drift between computed position and actual gripper location
- **If Issues**: Check transformation matrix multiplication order, verify T01-T56 are correct
- **Note**: This step can be skipped if you're confident in your FK implementation

---

### Large Sprint 2: The Referee (Collision Detection)
**Goal**: Implement collision detection and verify correctness through simulation testing.

#### Micro-Sprint 2.1: Point-in-AABB Function
- **Task**: Implement basic point-in-box test
- **Deliverable**:
  ```python
  def is_point_in_aabb(point: np.ndarray, aabb: AABB) -> bool:
      return np.all(point >= aabb.min) and np.all(point <= aabb.max)
  ```
- **Simple Test Script**:
  ```python
  # Test obvious cases, print results
  assert is_point_in_aabb([0.5, 0, 0.25], OBSTACLE_WALL) == True
  assert is_point_in_aabb([1.0, 0, 0], OBSTACLE_WALL) == False
  print("Point-in-AABB tests PASSED")
  ```

#### Micro-Sprint 2.2: Collision Detection Implementation
- **Task**: Implement simplified collision checker (end-effector only)
- **Deliverable**:
  ```python
  def is_in_collision(q: np.ndarray, obstacles: List[AABB]) -> bool:
      """Check if end-effector position collides with any obstacle."""
      ee_position = get_end_effector_position(q)
      for obstacle in obstacles:
          if is_point_in_aabb(ee_position, obstacle):
              return True
      return False
  ```
- **Test**: Call with configurations where end-effector is inside/outside obstacle
- **Validation**: Returns True when end-effector inside obstacle, False otherwise
- **Note**: This simplified approach allows robot arm joints to pass through obstacles

#### Micro-Sprint 2.3: Visual Collision Testing
- **Task**: In Webots, color-code configurations by collision status
- **Method**:
  - Test 10 configurations: 5 obviously safe, 5 obviously colliding
  - Draw end-effector sphere in **green** if `is_in_collision() == False`
  - Draw end-effector sphere in **red** if `is_in_collision() == True`
- **Deliverable**: Visual confirmation that collision detection is working
- **Success Criteria**: All obviously-colliding configs (end-effector inside obstacle) show red, safe configs show green
- **Debug**: If wrong, check AABB boundaries or FK implementation

#### Micro-Sprint 2.4: Boundary Testing
- **Task**: Test configurations where end-effector is very close to obstacle (near-miss cases)
- **Test Cases**:
  - End-effector just barely outside obstacle boundary (should be green/safe)
  - End-effector just barely inside obstacle boundary (should be red/collision)
  - End-effector far from obstacle (should be green/safe)
- **Method**: Manually create these configs, visualize with color coding
- **Validation**: Near-miss outside should be green, just inside should be red
- **Note**: Since only end-effector is checked, robot arm can pass through obstacle

---

### Large Sprint 3: The Target
**Goal**: Calculate and validate the goal configuration q_goal using IK.

#### Micro-Sprint 3.1: Define Target Pose
- **Task**: Add bin location to `config.py`
- **Deliverable**:
  ```python
  # config.py
  BIN_POSITION = np.array([0.5, 0.3, 0.4])  # XYZ in meters
  BIN_APPROACH_ORIENTATION = [0, 0, 0]  # Axis-angle or keep gripper pointing down
  ```
- **Validation**: Verify bin actually exists at this location in Webots world

#### Micro-Sprint 3.2: Extract/Adapt IK from hw2_code
- **Task**: Copy IK solver from `hw2_code/your_code_here.py` or re-implement
- **Options**:
  1. Import Newton-Raphson IK from hw2
  2. Use analytical UR5e IK if available
  3. Use PyBullet or ikpy library with UR5e URDF
- **Deliverable**: Function `inverse_kinematics(target_pose) -> Optional[np.ndarray]`
- **Test**: IK round-trip test - for random q, compute pose = FK(q), then q2 = IK(pose), verify q ≈ q2
- **Validation**: Print test results, ensure IK converges for reachable poses

#### Micro-Sprint 3.3: Calculate q_goal
- **Task**: Compute joint configuration to reach bin
- **Deliverable**:
  ```python
  # In your_code_here.py or config.py
  q_goal = inverse_kinematics(BIN_POSITION, desired_orientation)
  print(f"Goal configuration: {q_goal}")
  ```
- **Checks**:
  - All joint angles within UR5e limits (typically ±2π)
  - IK did not return None (failure)
  - No NaN values

#### Micro-Sprint 3.4: Visual q_goal Verification
- **Task**: Command Webots robot to move to q_goal
- **Method**: In controller, temporarily set motors to q_goal and run simulation
- **Observe**:
  - End effector is above/at bin location
  - No weird joint twisting or gimbal lock
  - Configuration looks natural and reachable
- **Success Criteria**: Visual inspection confirms correct placement

#### Micro-Sprint 3.5: q_goal Collision Check
- **Task**: Verify q_goal is collision-free
- **Test**:
  ```python
  assert not is_in_collision(q_goal, [OBSTACLE_WALL])
  print("q_goal is collision-free")
  ```
- **If Fails**: Adjust bin position or approach orientation, recompute IK, repeat
- **Validation**: Test passes before proceeding

---

### Large Sprint 4: The Planner (Math Only)
**Goal**: Implement RRT algorithm, test in pure Python without Webots visualization.

#### Micro-Sprint 4.1: RRT Node Structure
- **Task**: Define tree node class and tree storage
- **Deliverable**: In `rrt_planner.py`:
  ```python
  class RRTNode:
      def __init__(self, config: np.ndarray, parent: Optional['RRTNode'] = None):
          self.config = config
          self.parent = parent
  ```
- **Reference**: hw3_code/pygame_your_code_here.py uses lists for nodes/parents
- **Validation**: Create a few mock nodes, verify parent pointers work

#### Micro-Sprint 4.2: Configuration Sampling
- **Task**: Implement random sampling with goal biasing
- **Deliverable**:
  ```python
  def sample_configuration(q_goal: np.ndarray,
                          goal_bias: float = 0.1,
                          joint_limits: List[Tuple[float, float]]) -> np.ndarray:
      if np.random.random() < goal_bias:
          return q_goal.copy()
      # Sample uniformly within joint limits
      return np.array([np.random.uniform(low, high) for low, high in joint_limits])
  ```
- **Test**: Sample 1000 configs, count how many equal q_goal, should be ~100 (10%)
- **Validation**: Print statistics, verify bias percentage

#### Micro-Sprint 4.3: Nearest Neighbor Search
- **Task**: Find closest tree node to sampled configuration
- **Deliverable**:
  ```python
  def nearest_node(nodes: List[RRTNode], q_sample: np.ndarray) -> RRTNode:
      return min(nodes, key=lambda node: np.linalg.norm(node.config - q_sample))
  ```
- **Note**: For small trees (<1000 nodes), linear search is fine. For larger, consider scipy KDTree
- **Test**: Create tree with 5 known nodes, verify correct nearest is found
- **Validation**: Print test results

#### Micro-Sprint 4.4: Steering Function
- **Task**: Step toward sample by limited distance
- **Deliverable**:
  ```python
  def steer(q_from: np.ndarray, q_to: np.ndarray,
            step_size: float, joint_limits: List[Tuple]) -> np.ndarray:
      direction = q_to - q_from
      distance = np.linalg.norm(direction)
      if distance <= step_size:
          return q_to
      q_new = q_from + (direction / distance) * step_size
      # Clamp to joint limits
      return np.clip(q_new, [low for low,_ in joint_limits],
                            [high for _,high in joint_limits])
  ```
- **Reference**: hw3_code uses `utils.steer()` with same logic
- **Test**: Steer from [0,0,0,0,0,0] toward [1,1,1,1,1,1] with step=0.5, verify result
- **Validation**: Print result, check it's on line segment and distance ≤ step_size

#### Micro-Sprint 4.5: Path Extraction
- **Task**: Trace parent pointers from goal to start
- **Deliverable**:
  ```python
  def extract_path(goal_node: RRTNode) -> List[np.ndarray]:
      path = []
      current = goal_node
      while current is not None:
          path.append(current.config)
          current = current.parent
      return list(reversed(path))
  ```
- **Test**: Create mock tree with known path, verify extraction
- **Validation**: Path[0] is start, Path[-1] is goal

#### Micro-Sprint 4.6: Core RRT Loop (Without Collision)
- **Task**: Implement main planning loop, stub out collision checking
- **Deliverable**:
  ```python
  class RRTPlanner:
      def plan(self, q_start, q_goal, max_iterations=5000):
          nodes = [RRTNode(q_start)]
          for i in range(max_iterations):
              q_sample = self.sample_configuration(q_goal)
              q_nearest_node = self.nearest_node(nodes, q_sample)
              q_new = self.steer(q_nearest_node.config, q_sample)
              # TODO: collision check (assume valid for now)
              nodes.append(RRTNode(q_new, parent=q_nearest_node))
              if np.linalg.norm(q_new - q_goal) < goal_threshold:
                  return extract_path(nodes[-1])
          return None
  ```
- **Test**: Run planner with no obstacles, should find path quickly
- **Validation**: Print "Path found! Nodes: X, Iterations: Y"

#### Micro-Sprint 4.7: Integrate Collision Checking
- **Task**: Add collision check before adding node to tree
- **Modification**:
  ```python
  if not is_in_collision(q_new, obstacles):
      nodes.append(RRTNode(q_new, parent=q_nearest_node))
  ```
- **Test**: Place obstacle between start and goal, verify planner finds path around it
- **Validation**: Run simulation, print progress every 100 iterations

#### Micro-Sprint 4.8: Console-Only Planner Testing
- **Task**: Run RRT planner without Webots, just print results
- **Test Scenarios**:
  1. No obstacle: Should succeed in <500 iterations
  2. Obstacle blocking direct path: Should find path around (may take longer)
  3. q_goal surrounded by obstacle (impossible): Should timeout and return None
- **Deliverable**: Script that runs these tests and prints statistics
- **Validation**: All three scenarios behave as expected

#### Micro-Sprint 4.9: Parameter Tuning
- **Task**: Experiment with step_size, max_iterations, goal_threshold
- **Method**: Run planner multiple times with different parameters, log results
- **Metrics**: Success rate (% of runs that find path), iterations needed, planning time
- **Experiments**:
  - step_size: [0.05, 0.1, 0.2, 0.5] - test each with 10 runs
  - goal_threshold: [0.1, 0.2, 0.5] - test with best step_size
- **Deliverable**: Update `config.py` with optimal parameters
- **Validation**: Document findings: "step_size=0.1 gives 90% success, avg 1500 iterations"

---

### Large Sprint 5: Execution
**Goal**: Execute planned paths on the Webots robot.

#### Micro-Sprint 5.1: Path Validation
- **Task**: Verify entire path is collision-free before execution
- **Deliverable**:
  ```python
  def validate_path(path: List[np.ndarray], obstacles: List[AABB]) -> bool:
      for q in path:
          if is_in_collision(q, obstacles):
              print(f"WARNING: Path contains collision at config {q}")
              return False
      return True
  ```
- **Test**: Test with known-good and known-bad paths
- **Validation**: Correctly identifies invalid paths

#### Micro-Sprint 5.2: Linear Path Interpolation
- **Task**: Densify path for smooth motion
- **Deliverable**: In `path_utils.py`:
  ```python
  def interpolate_path(path: List[np.ndarray], resolution: float = 0.05) -> List[np.ndarray]:
      """Add waypoints every `resolution` radians along path."""
      dense_path = [path[0]]
      for i in range(len(path) - 1):
          q_from, q_to = path[i], path[i+1]
          distance = np.linalg.norm(q_to - q_from)
          num_steps = max(int(distance / resolution), 1)
          for step in range(1, num_steps + 1):
              alpha = step / num_steps
              q_interp = q_from + alpha * (q_to - q_from)
              dense_path.append(q_interp)
      return dense_path
  ```
- **Test**: Interpolate between 2 waypoints, verify smooth spacing
- **Validation**: Print waypoint count before/after interpolation

#### Micro-Sprint 5.3: Optional Cubic Trajectory Smoothing
- **Task**: Use cubic polynomials for even smoother motion
- **Reference**: hw3_code/webots_your_code_here.py has `get_cubic_s(t, T)`
- **Deliverable**:
  ```python
  def smooth_segment(q_start, q_end, num_steps):
      """Generate smooth trajectory using cubic polynomial."""
      trajectory = []
      for i in range(num_steps):
          s = get_cubic_s(i, num_steps)  # From hw3
          q = q_start + s * (q_end - q_start)
          trajectory.append(q)
      return trajectory
  ```
- **Validation**: Visualize velocity profile, should start/stop at zero velocity

#### Micro-Sprint 5.4: Webots Motor Controller
- **Task**: Command UR5e motors to follow path waypoints
- **Deliverable**: In `your_code_here.py`:
  ```python
  def execute_path(robot, motors, path):
      for q in path:
          for i, motor in enumerate(motors):
              motor.setPosition(q[i])
          robot.step(TIME_STEP)
          # Optional: wait until position reached or fixed duration
  ```
- **Test**: Execute a 3-waypoint path, observe robot motion
- **Validation**: Robot moves to each waypoint in sequence

#### Micro-Sprint 5.5: End-to-End Integration
- **Task**: Full pipeline from planning to execution
- **Deliverable**: Main function in `your_code_here.py`:
  ```python
  def main():
      # 1. Define start (current robot config) and goal
      q_start = get_current_joint_angles()
      q_goal = compute_goal_config()

      # 2. Plan path
      planner = RRTPlanner()
      path = planner.plan(q_start, q_goal, obstacles)
      if path is None:
          print("Planning failed!")
          return

      # 3. Validate and interpolate
      assert validate_path(path, obstacles)
      dense_path = interpolate_path(path)

      # 4. Execute
      execute_path(robot, motors, dense_path)
      print("Path execution complete!")
  ```
- **Test**: Run complete pipeline in Webots
- **Success Criteria**: Robot moves from start to goal without hitting obstacle

#### Micro-Sprint 5.6: Path Visualization Before Execution
- **Task**: Draw planned path in Webots before moving robot
- **Method**: Use Supervisor to draw end-effector position (sphere) for each waypoint in green
- **Purpose**: Final visual check before committing to motion - trace shows end-effector trajectory
- **Deliverable**: Add visualization step between planning and execution
- **Validation**: Path appears reasonable and end-effector positions avoid obstacles

#### Micro-Sprint 5.7: Error Handling
- **Task**: Add graceful failure handling
- **Cases**:
  - Planning fails (returns None) → log error, don't crash
  - Path validation fails → reject path, alert user
  - IK fails for q_goal → try alternative approach or report
- **Deliverable**: Try-except blocks with informative messages
- **Validation**: Trigger each error scenario, verify graceful handling

---

### Large Sprint 6: Evaluation
**Goal**: Run systematic experiments and document performance.

#### Micro-Sprint 6.1: Experiment Framework
- **Task**: Create script to run multiple trials with varying parameters
- **Deliverable**: `experiments.py`:
  ```python
  def run_experiment(step_size, obstacle_config, num_trials=30):
      results = []
      for trial in range(num_trials):
          start_time = time.time()
          path = planner.plan(q_start, q_goal)
          planning_time = time.time() - start_time

          results.append({
              'trial': trial,
              'success': path is not None,
              'planning_time': planning_time,
              'path_length': len(path) if path else 0,
              'num_nodes': planner.get_tree_size()
          })
      return results
  ```
- **Output**: Save results to CSV file
- **Validation**: Run 5 trials, verify CSV is generated correctly

#### Micro-Sprint 6.2: Baseline Performance
- **Task**: Run 30 trials with default parameters
- **Config**: Fixed obstacle, fixed start/goal, step_size=0.1
- **Metrics**: Success rate, avg planning time, avg path length
- **Deliverable**: `results_baseline.csv` and summary statistics
- **Validation**: Compute mean/std for all metrics, success rate should be >80%

#### Micro-Sprint 6.3: Step Size Sensitivity
- **Task**: Vary step_size while keeping everything else constant
- **Values**: [0.05, 0.1, 0.2, 0.5] × 20 trials each
- **Hypothesis**: Smaller step_size → slower planning, smoother paths
- **Deliverable**: Comparative results CSV
- **Analysis**: Plot planning time vs step_size, identify optimal value
- **Validation**: Observe expected trade-offs in data

#### Micro-Sprint 6.4: Obstacle Position Variation
- **Task**: Move obstacle to different locations, test robustness
- **Scenarios**:
  1. Obstacle directly between start/goal (hardest)
  2. Obstacle offset to side (medium)
  3. Obstacle far away (easy)
- **Trials**: 15 runs per scenario
- **Deliverable**: Success rate comparison across scenarios
- **Validation**: Harder scenarios have lower success rate or longer planning time

#### Micro-Sprint 6.5: Visualization and Plots
- **Task**: Generate plots summarizing results
- **Plots**:
  - Success rate vs step_size (bar chart)
  - Planning time distribution (histogram)
  - Path length vs obstacle position (box plot)
  - Success rate vs scenario difficulty (bar chart)
- **Tools**: matplotlib or similar
- **Deliverable**: `plots/` directory with PNG images
- **Validation**: Plots clearly show trends and differences

#### Micro-Sprint 6.6: Final Documentation
- **Task**: Write summary of findings
- **Deliverable**: Update README.md or create `results_summary.md`
- **Sections**:
  - Methodology (what you tested)
  - Results (key metrics with tables)
  - Discussion (what worked, what didn't, why)
  - Limitations (known issues, edge cases)
  - Future work (RRT*, bidirectional RRT, dynamic obstacles)
- **Validation**: Another person could understand your project from the docs

---

## AI Agent Collaboration Notes

### When Working on This Project
- **STOP after EACH micro-sprint**: Show results, ask "Are you satisfied? Should I proceed?", WAIT for user approval
- **Read existing code first**: Check hw2_code, hw3_code, hw4_code for reusable implementations
- **Follow sprint order**: Each sprint builds on previous, don't skip ahead
- **Visualize everything**: When in doubt, draw it in Webots
- **Test incrementally**: Small changes, test immediately, don't accumulate untested code
- **Controller management**: If you change .wbt controller for testing, REMEMBER to change it back afterward
- **Use print statements liberally**: Better than silent failures

---

Remember: **Test small, test often, visualize everything.**
