import arcade
import numpy as np
from numpy.typing import NDArray
import random


AU_p_Pix=1/50
G = 39.478    # AU/M⊙/year
solarMass=1.988416e30   # 1 Solar Mass = 1.988416e30
softening = 1e-5
global_center_x=0
global_center_y=0


positions = np.empty((0, 2), dtype=np.float64)      # Pixels ,  Distances in Astronmical Unit
velocities = np.empty((0, 2), dtype=np.float64)
accelerations = np.empty((0, 2), dtype=np.float64)    #not nessecary just incase for later use
masses= np.empty(0,dtype=np.float64)        #Mass in Solar Mass unit

store_velocities=list(np.empty((0, 2), dtype=np.float64))
store_positions=list(np.empty((0, 2), dtype=np.float64))

"""Base Object"""
class Celestial(arcade.SpriteCircle):

    def __init__(self, Name : str ,
                 index:int,
                 radius : int,
                 position:tuple[float,float],
                 color ):
        
        super().__init__(radius,color)
                
        self.name=Name
        self.index=index
        
        self.radius=radius
        self.color=color
        
        self.center_x=position[0]
        self.center_y=position[1]
    
    def getDistanceFromOther(self, other:"Celestial") -> np.float64:
        return np.float64(np.linalg.norm(positions[self.index] - positions[other.index]))
    
    def update_texture(self,radius,color) -> None:
        self.texture = arcade.make_circle_texture(radius * 2, color)
        self.width = radius * 2
        self.height = radius * 2
        
        
"""" Every Object will be through addCelestial"""
def addCelestial(spriteList:arcade.SpriteList,
                 Name : str ="",
                 radius : int = 50,
                 position : tuple[float,float] = (0,0),
                 mass : float = 0,
                 velocity : tuple[float,float] = (0,0),
                 acceleration : tuple[float,float] = (0,0),
                 color = arcade.color.WHITE) -> None:
    
    global positions, velocities, accelerations, masses

    index=len(masses)
    
    # We are adding global center as the input values are from screen's center (0,0) but arcade screen starts from bottem left
    # WE are multiplying AU_p_Pix as the input values are in AU but Global Center in pixels    
    position=(position[0]+(global_center_x*AU_p_Pix),position[1]+(global_center_y*AU_p_Pix))
        
    if Name=="" :
        Name="Planet"+str(index) 
    
    # We are sending positions in Celestial in Pixels as it has to be used for display. Positions in in AU
    sprite=Celestial(Name,index,radius,(position[0]/AU_p_Pix,position[1]/AU_p_Pix),color)
    spriteList.append(sprite)
    
    positions = np.vstack([positions, np.array([position])])
    # We converted Pixels in to Au just here so no need to convert when calculation but AU to Pix when displaying
    
    velocities = np.vstack([velocities, np.array([velocity])])
    accelerations = np.vstack([accelerations, np.array([acceleration])])
    masses = np.append(masses, mass)
    
def deleteCelestial(spriteList:arcade.SpriteList,sprite:Celestial) -> None:
    global positions,velocities,accelerations,masses
    
    index = sprite.index
    
    spriteList.remove(sprite)
    positions = np.delete(positions, index, axis=0)
    velocities = np.delete(velocities, index, axis=0)
    accelerations = np.delete(accelerations, index, axis=0)
    masses = np.delete(masses,index)


class myEnvironment(arcade.Window):
    def __init__(self,title=""):
        
        global global_center_x, global_center_y
        
        self.is_fullscreen=False
        
        screen_w, screen_h = arcade.get_display_size()
        print(f"Resolution: {screen_w}x{screen_h}")

        self.window_w = int(screen_w * 0.8)
        self.window_h = int(screen_h * 0.73)

        super().__init__(self.window_w, self.window_h,title)
        self.set_location(0, 38)
        
        global_center_x=self.center_x
        global_center_y=self.center_y


    def setup(self):
        arcade.set_background_color(arcade.color.BLACK)
        
        self.camera = arcade.Camera2D()
        self.dragging=False
        self.paused=False
        self.merger=True
        self.print_Debug=False
        
        self.planets=arcade.SpriteList()
        # KELPER 35
        # addCelestial(self.planets,Name="Planet1",radius=20,position=(-0.084,0),mass=0.8877,velocity=(0,-10.1978))
        # addCelestial(self.planets,"Planet2",20,(0.092,0),0.8094,(0,+9.3025))
        # SUN & EARTH
        # addCelestial(self.planets,Name="Sun",radius=80,position=(2.95e-6,0),mass=1,velocity=(0,+1.917e-5))
        # addCelestial(self.planets,Name="Earth",radius=20,position=(-0.983,0),mass=3.003e-6,velocity=(0,-6.386))
        # Amazing example of 3 body
        addCelestial(self.planets,Name="Sun",radius=8,position=(2.95e-6,0),mass=1,velocity=(0,+1.917e-5))
        addCelestial(self.planets,Name="Earth",radius=2,position=(0.983,0),mass=3.003e-6,velocity=(0,-6.386))
        addCelestial(self.planets,Name="",radius=5,position=(2,1),mass=6e-1,velocity=(2,-2))
        # SUN, Earth, Jupiter
        # addCelestial(self.planets,Name="Sun",radius=8,position=(0.004966,0),mass=1,velocity=(0,0.0026))
        # addCelestial(self.planets,Name="Earth",radius=2,position=(0.983,0),mass=3.003e-6,velocity=(0,-6.386))
        # addCelestial(self.planets,Name="Jupiter",radius=4,position=(5.20,0),mass=9.545e-4,velocity=(0,-2.755))
        # addCelestial(self.planets,Name="Jupiter",radius=4,position=(5.,-3),mass=1.5,velocity=(-3,-2.5))

        # for i in range(-2,2):
        #     for j in range(-2,2):
        #         addCelestial(self.planets,"",radius=3,position=(i,j),mass=random.randrange(1,3),velocity=(random.randrange(0,2),random.randrange(0,2)))
        
        xs=positions[:,0]/AU_p_Pix
        ys=positions[:,1]/AU_p_Pix
        self.trail_list=[]
        self.trail_list.append(list(zip(xs,ys)))
        
        self.deleted=np.empty(0,dtype=int)
        
        self.v_dist=np.empty(0,dtype=np.float64)
        self.r_dist=np.empty(0,dtype=np.float64)
        
                
        self.dt=1   # 1 year is kept as default
        self.frame_count=0
        self.time_elapsed=0
        self.next_snapshot=0
        
        self.trail_list=[[] for _ in range(len(positions))]


    def on_draw(self):
        global positions
        self.clear()
        self.camera.use()
        
        color=[arcade.color.RED,arcade.color.BLUE,arcade.color.GREEN,arcade.color.YELLOW,arcade.color.MAGENTA,arcade.color.PINK,arcade.color.PURPLE]
        for i,j in enumerate(self.trail_list):
            arcade.draw_line_strip(j,color[i%len(color)])
        self.planets.draw()
        
                     
    def on_update(self,delta_time):
        
        if not self.paused:
            
            
            
            global positions, velocities, masses, accelerations,store_velocities,store_positions
            
            if self.print_Debug:
                if self.frame_count%30==0:
                    print("Position: ",positions)
                    print("velocity: ",velocities)
                    print("masses: ",masses)
                    # print(self.Tidal_perturabation(self.planets[0]))
            
            
            # if self.frame_count%20==0:
            #     print(np.sum(masses[:, None] * velocities, axis=0))
        
        
        
            
            
            if self.merger:
                self.collision_handler()

            self.update_time_period()
            

            """Calculate Acceleration, Velocity and Position"""            
            acc_curr = self.cal_Acceleration()
            # Get the current acceleration due to gravity
            
            positions= positions+ velocities*self.dt + 0.5*acc_curr*self.dt**2 
            # Calculate the new positions of Objects
            
            positions[self.deleted]=(0,0)
            
            acc_new = self.cal_Acceleration()  # <-- this is a(t+dt)  required to cal Velocity
            # Get acceleration after a period of time stamp has passed
            
            velocities = velocities + 0.5*(acc_curr + acc_new)*self.dt
            # Calculates the new velocity
            
            velocities[self.deleted]=(0,0)
            

            """Setting new positions"""
            xs = positions[:, 0]/AU_p_Pix
            ys = positions[:, 1]/AU_p_Pix
            for sprite in self.planets:

                sprite.center_x = xs[sprite.index]
                sprite.center_y = ys[sprite.index]


            """Storing data for trail: will only have data for specific number of frames, bluring trails"""
            self.frame_count+=1
            if self.frame_count%2 == 0:
                
                scaled_positions = (positions / AU_p_Pix).astype(np.float16)
                for trail, pos in zip(self.trail_list, map(tuple, scaled_positions)):
                    if pos==(0,0):
                        continue
                    trail.append(pos)
            
                            
                    
            self.time_elapsed+=self.dt
            if self.time_elapsed >= self.next_snapshot:
                store_positions.append(positions)
                store_velocities.append(velocities)
                self.next_snapshot += 10

                        


    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.F:
            self.is_fullscreen = not self.is_fullscreen
            self.set_fullscreen(self.is_fullscreen)

            if not self.is_fullscreen:
                self.set_size(self.window_w, self.window_h)
                self.set_location(0, 38)

            global global_center_x, global_center_y
            global_center_x=self.center_x
            global_center_y=self.center_y
            
        elif symbol == arcade.key.SPACE:
            self.paused= not self.paused
            
        elif symbol == arcade.key.P:
            self.print_Debug = not self.print_Debug

    def on_mouse_press(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            self.dragging = True
            self.last_mouse_pos = (x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            self.dragging = False

    def on_mouse_motion(self, x, y, dx, dy):
        if self.dragging:
            # Invert dx/dy so dragging feels like "grabbing" the scene
            self.camera.position = (
                self.camera.position[0] - dx,
                self.camera.position[1] - dy
            )
    
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        move_speed = 50  # in pixels
        zoom_speed = 0.1
        if scroll_y > 0:
            self.camera.zoom *= (1 + zoom_speed)
        elif scroll_y < 0:
            self.camera.zoom /= (1 + zoom_speed)

    def cal_Acceleration(self): 
        """
        # COMMENTED IS Code FOR LOGIC OF WHATS HAPPENING
        N = positions.shape[0]
        acc = np.zeros_like(positions)
        for i in range(N):            
            diff=positions[i] - positions       
            distances=np.linalg.norm(diff,axis=1)
            inv_dist3 = 1.0 / ((distances)**2 + softening**2)**1.5
            acc[i] = -G * np.sum((diff.T * masses * inv_dist3).T, axis=0)
        """
        
        diffSqred = (positions[:, np.newaxis, :] - positions[np.newaxis, :, :])
        # [[(x1-x2,y1-y2)]]
        dist = np.sum((diffSqred)**2, axis=2)
        # (x1-x2)^2 + (y1-y2)^2
        inv_dist3 = (dist+softening**2) **-1.5  #as per formula this shoud be 3 but in above part we didn't square root and compensentiated here
        # 1 / ( r + softening ^ 2 )^(3/2)     r is supposed to be squared but when taking eular's distance we take underroot, both cancels out 
        np.fill_diagonal(inv_dist3, 0.0)
        # Removes the Infinity from the matrix
        acc = -G * (diffSqred* inv_dist3[:,:,None] * masses[None,:,None]).sum(axis=1)
        # Adds extra Dimentions in inv_dist3 and masses so that it can match diff
        return acc

    """Merge OverLapping Objects / Handle Collisions"""
    def collision_handler(self) ->None:
        for i,sprite in enumerate(self.planets):
            hit = arcade.check_for_collision_with_list(sprite,self.planets,method=0) # type: ignore
            if hit:
                for obj in hit:
                    big,small = (sprite,obj) if masses[sprite.index] > masses[obj.index] else (obj,sprite)
                    
                    mass_b=masses[big.index]
                    mass_s=masses[small.index]
                    
                    velocities[big.index] = (mass_b*velocities[big.index]+mass_s*velocities[small.index,None])/(mass_b+mass_s)
                    positions[big.index] = (mass_b*positions[big.index]+mass_s*positions[small.index,None])/(mass_b+mass_s)
                    masses[big.index] += mass_s
                    
                    velocities[small.index] = (0,0)
                    positions[small.index] =(0,0)
                    masses[small.index] = 0
                    
                    big.update_texture(int(((big.radius**3)+(small.radius**3))**(1/3)),big.color)
                    
                    self.deleted=np.append(self.deleted,small.index)
                    self.planets.remove(small)

    """Euler's Distance"""
    def get_distance(self,coordinate_list:np.ndarray):
        diffs = coordinate_list[:, np.newaxis, :] - coordinate_list[np.newaxis, :, :]
        temp= np.linalg.norm(diffs, axis=2)
        temp[self.deleted]=0
        temp[:,self.deleted]=0
        return temp
        
    """Get dynamic time period"""
    
    def update_time_period(self)->None:
        
        # self.dt = max(dt_min, min(dt_max, safety_factor * d_min / v_max))
        dt_min_possible=0.0001
        dt_max_possible=20
        
        dist=self.get_distance(positions)
        dist[dist==0]=np.inf    #ignore self distance and deleted planets
        min_dist = dist.min()
        speeds = np.linalg.norm(velocities, axis=1)   # speed of each object
        max_velo = speeds.max()   
        self.dt = max(dt_min_possible, min(dt_max_possible, 0.2 * min_dist / max_velo))/3
    
        
    """ Specific Orbital Energy (ε) of every object relative to each other """
    # Used to check if two planets are bound to each other or are being thorwn away RELATIVE TO EACH OTHER ONLY (Other planets may have a effect)
    def Orbital_energy(self):
        v=self.get_distance(velocities)
        r=self.get_distance(positions)
        
        e=(v**2)/2 - G*(masses[:,np.newaxis]+masses[np.newaxis,:])/ r
        return e
    
    """Angular Momentum of Orbits"""
    def angular_momentum(self):
        return np.cross(positions,velocities)*masses
    
    """Pair wise Energy: are bodies gravitationally bound to each other (Is the system stable)"""
    def PairwiseEnergy(self):
        mul_mi_mj = masses[:,np.newaxis]*masses[np.newaxis,:]
        add_mi_mj = masses[:,np.newaxis]+masses[np.newaxis,:]
        v_dist_sqrd=self.get_distance(velocities)**2
        r_dist=self.get_distance(positions)
        
        E=0.5*(mul_mi_mj/add_mi_mj)*v_dist_sqrd-G*(mul_mi_mj/r_dist)
        return E    
    
    """Tidal perturbation ratio 
    
       [[nan  nan nan]   # primary (Sun) row ignored
        [nan  0   1  ]   # body 1 (say Earth)
        [nan  2   0  ]]  # body 2 (say Jupiter)
    
        .) Matrix[1,2] → effect of body 2 on body 1 (Jupiter's perturbation on Earth).
        .) Matrix[2,1] → effect of body 1 on body 2 (Earth's perturbation on Jupiter).
        .) Matrix[i,i] = 0 (no self-perturbation).
    """
    # WILL HAVE TO CHECK THE LINE WITH ALL nan'S AND IGNORE IT 
    # AS IT IS THE FORCE PRIMARY'S FORCE EFFECT ON ITSELF AND AS PER FORMULA IT DOSENT MAKES SENSE

    def Tidal_perturabation(self,primary:Celestial):
        r_dist=np.linalg.norm(positions - positions[primary.index], axis=1)
        n=(masses/masses[primary.index])*((r_dist[:, None]/r_dist[None, :]))**3
        n[primary.index] = np.nan
        n[:,primary.index] = np.nan
        np.fill_diagonal(n,np.nan)
        return n
        


    
Env=myEnvironment(title="First")
Env.setup()

arcade.run()