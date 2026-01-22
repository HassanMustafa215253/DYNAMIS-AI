import taichi as ti
from typing import Optional

AU_p_radius=1/50

MAX_PARTICLES: Optional[ti.Field]
MAX_COLLISIONS: Optional[ti.Field]
positions: Optional[ti.Field] 
velocities: Optional[ti.Field]
acceleration: Optional[ti.Field]
mass: Optional[ti.Field]
radii: Optional[ti.Field]
collision_count: Optional[ti.Field]
collisions: Optional[ti.Field]
ret: Optional[ti.Field]
ret_idx: Optional[ti.Field]
change: Optional[ti.Field]
grid: Optional[ti.Field] =None
grid_count: Optional[ti.Field] =None
grid_min_x: Optional[int]
grid_min_y: Optional[int]
bounds: Optional[ti.Field]
minx=10000000
maxx=0
miny=10000000
maxy=0


ti.init(arch=ti.gpu)
grid_min_x=ti.field(ti.i32, shape=())
grid_min_y=ti.field(ti.i32, shape=())

cell_size = ti.field(ti.f32, shape=())


# output


def bind_buffers(pos, m, vel, acc, rad,ret_arr,idx,checkChange,boundss):
    global positions, mass, velocities, acceleration, radii,ret, ret_idx, change
    global MAX_PARTICLES, MAX_COLLISIONS, collision_count, collisions, bounds
    positions = pos
    mass = m
    velocities = vel
    acceleration = acc
    radii = rad
    ret = ret_arr 
    ret_idx = idx
    change = checkChange
    bounds=boundss
    
    MAX_PARTICLES = positions.shape[0]        # upper bound
    MAX_COLLISIONS = positions.shape[0]*10      # safety cap
    
    collision_count = ti.field(ti.i32, shape=())
    collisions = ti.Vector.field(2, ti.i32, shape=MAX_COLLISIONS)
    
    
    
    
    

def collision_manager():
    
    global grid, grid_count, grid_min_x, grid_min_y, bounds
    global cell_size, grid_width, grid_height, grid_size
    
    # determine bounds
    
    assert positions != None, "bind_buffers() must be called first"
    assert velocities != None, "bind_buffers() must be called first"
    assert mass != None, "bind_buffers() must be called first"
    assert acceleration != None, "bind_buffers() must be called first"
    assert radii != None, "bind_buffers() must be called first"

    """We are using ret_idx to pass the biggest radius of particles as a shortcut
       else we would need to make another variable for it or calculate it again here
       that would require to convert to numpy then use max() which is slow"""
    cell_size[None] = ret_idx[None]*2
    
    """We are resetting ret_idx to zero in Find_Collisions kernel after counting collisions so no need to reset here"""
    
    # spatial grid
    # unpack for clarity

    # required grid size (in cells)
    if (bounds[0].x < grid_min_x or bounds[0].y > grid_min_x or
    bounds[1].x < grid_min_y or bounds[1].y > grid_min_y):
    req_width  = int((bounds[0].y - bounds[0].x) / cell_size[None]) + 1
    req_height = int((bounds[1].y - bounds[1].x) / cell_size[None]) + 1

    # over-allocate (grow factor)
    GROWTH = 50
    grid_width  = req_width  * GROWTH
    grid_height = req_height * GROWTH

    # center the grid in world space
    center_x = 0.5 * (bounds[0].x + bounds[0].y)
    center_y = 0.5 * (bounds[1].x + bounds[1].y)

    grid_min_x = center_x - 0.5 * grid_width  * cell_size[None]
    grid_min_y = center_y - 0.5 * grid_height * cell_size[None]
    # reallocate ONLY if current grid is too small
    if grid_count == None or grid_width > grid_count.shape[0] or grid_height > grid_count.shape[1]:
        print("orignal grid size:", grid_count.shape[0] if grid_count != None else "N/A", grid_count.shape[1] if grid_count != None else "N/A")
        print("grid resized:", grid_width, grid_height)
        grid_count = ti.field(ti.i32, shape=(grid_width, grid_height))
        print("limits x:", bounds[0].y, bounds[0].x)
        print("limits y:", bounds[1].y, bounds[1].x)
        grid       = ti.field(ti.i32, shape=(grid_width, grid_height, 64))  # max 64 per cell

    build_grid()
    find_collisions()
    
    
@ti.kernel
def build_grid():
    # print(1)
    for i, j in grid_count:
        grid_count[i, j] = 0
    # print(2)
    for i in range(MAX_PARTICLES):
        if mass[i] ==0:
            continue
        cx = int((positions[i].x - grid_min_x) / cell_size[None])
        cy = int((positions[i].y - grid_min_y) / cell_size[None])

        if cx < 0 or cx >= grid_width or cy < 0 or cy >= grid_height:
            print("Out of bounds", cx, cy, positions[i], grid_min_x, grid_min_y,grid_width,grid_height)


        idx = ti.atomic_add(grid_count[cx, cy], 1)
        if idx < 64:
            grid[cx, cy, idx] = i
            
            
@ti.kernel
def find_collisions():
        
    collision_count[None] = 0

    for cx, cy in grid_count:
        for a in range(grid_count[cx, cy]):
            i = grid[cx, cy, a]

            for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
                nx = cx + dx
                ny = cy + dy

                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    for b in range(grid_count[nx, ny]):
                        j = grid[nx, ny, b]
                        if j <= i:
                            continue

                        d = positions[i] - positions[j]
                        r = radii[i]*AU_p_radius + radii[j]*AU_p_radius
                        # print("d", d)
                        # print("r", r,"\n\n")
                        if d.dot(d) <= r * r:
                            idx = ti.atomic_add(collision_count[None], 1)
                            if idx < MAX_COLLISIONS:
                                collisions[idx] = ti.Vector([i, j])
                                
    zero2 = ti.Vector([0.0, 0.0])  # define once outside the loop
    

    if collision_count[None] > 0:
        change[None] = 1

    ret_idx[None] = 0
    ti.loop_config(serialize=True)
    for i in range(collision_count[None]):
        a = collisions[i][0]
        b = collisions[i][1]

        # skip dead particles early
        if mass[a] == 0 or mass[b] == 0:
            continue

        # decide which is big/small
        big = small = 0  # initialize with dummy values
        if mass[a] >= mass[b]:
            big, small = a, b
        else:
            big, small = b, a

        mb = mass[big]
        ms = mass[small]

        total_mass = mb + ms
        
        
        
        
        # weighted averages
        velocities[big] = (mb * velocities[big] + ms * velocities[small]) / total_mass
        positions[big] = (mb * positions[big] + ms * positions[small]) / total_mass
        mass[big] = total_mass
        # print("positions:", positions[big],"    " ,positions[small])

        # deactivate small
        velocities[small] = (0.000000,0.000000)
        positions[small] = (0.000000,0.000000)
        mass[small] = 0.0
        
        # print(f"Merging particle {big} into {small}")
        # print("Positions")
        # for t in range(collision_count[None]):
        #     print("t:",t,"  ", positions[t])
        # print("velocities")
        # for t in range(collision_count[None]):
        #     print("t:",t,"  ", velocities[t])
        # print("masses")
        # for t in range(collision_count[None]):
        #     print("t:",t,"  ", mass[t])
        # print("\n\n")

        # print("positions:", positions[big],"    " ,positions[small])
        # store the index of deactivated particle
        # print("Collision detected between particles", small ," " ,ms, "and", big," " ,mb)
        ret[ret_idx[None]] = ti.Vector([small, big])
        ret_idx[None] += 1
