# Mapping, planning and navigation

The coordinate convention is world x/y with counterclockwise yaw and a body
x axis pointing forward. Wheel order is left then right. Positive differential
right-wheel speed produces positive yaw rate.

Occupancy grids index [row, column] = [y, x]. Cell centers have half-cell
offsets from the metric origin. Rasterized rectangles include touching cells,
so footprint inflation is intentionally conservative. The Euclidean distance
transform subtracts a half-cell diagonal to bound distance to occupied cell
area, and map boundaries behave as occupied space.

A-star uses octile distance and nonnegative traversal penalties. Diagonal
moves cannot cut blocked corners. Route shortcutting uses exact cell-boundary
supercover checks, including cells touched only at a corner; uniform sampling
alone can miss short intersections. Clearance-weighted costs favor open aisles.

The mission follows this fixed known-map route with arc-length lookahead and
pure pursuit. A progress floor prevents sensor noise from jumping the tracker
to a distant route segment. Forward-speed limits include stopping distance,
latency and an uncertainty margin. Short swept-footprint checks suppress
unsafe forward commands. Physical collisions terminate the simulation with
an error rather than allowing the robot to pass through obstacles.

The circular footprint has no orientation-dependent growth. Heading and
wheel acceleration constraints still affect motion. Pure pursuit and the
supervisory speed rule do not constitute a proof of recursive feasibility or
collision avoidance for arbitrary disturbances; test the complete scenario.

Additional planning tools include dynamic-window rollouts, heading lattices,
time-expanded node reservations, coverage sweeps, frontier utilities, small
exact multi-stop tours and nonnegative directed energy graphs.
Time-expanded reservations constrain occupied nodes only; callers must add
edge-conflict rules for multi-robot swap avoidance. Energy graphs require
nonnegative edges and therefore do not directly optimize signed regenerative
energy over arbitrary cycles.

No external robotics framework or middleware is required. There are no ROS
nodes, hardware drivers, online SLAM or dynamic-obstacle mission replanning.
The separate algorithms are available for experiments without implying that
every algorithm runs in the default warehouse pipeline.
