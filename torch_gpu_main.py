import arcade
import numpy as np
from numpy.typing import NDArray
import random
import CollisionDetectorVersions.pipitorch as wt
# import jax.numpy as jnp
# import jax
import torch

AU_p_Pix=1/50   # 1 AU = 50 pixels
AU_p_radius=1/50   # 1 AU = 2000 radius units
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
store_masses=[]

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
                 id_to_sprites:list,
                 Name : str ="",
                 radius : int = 1,
                 position : tuple[float,float] = (0,0),
                 mass : float = 0,
                 velocity : tuple[float,float] = (0,0),
                 acceleration : tuple[float,float] = (0,0),
                 color = arcade.color.WHITE) -> None:
    
    global positions, velocities, accelerations, masses

    index=len(masses)
    
    if Name=="" :
        Name="Planet"+str(index) 
    
    # we are sending position as (0,0) as we will set the actual position after adjusting barycenter in setup function of Environment class
    sprite=Celestial(Name,index,radius,(0,0),color)
    spriteList.append(sprite)
    id_to_sprites.append(sprite)
    
    positions = np.vstack([positions, np.array([position])])
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
        self.dynamic_dt=True
        
        self.planets=arcade.SpriteList()
        self.id_to_sprites=[]
        self.deleted=[]
        
        self.frame_count=0
        self.time_elapsed=0
        self.next_snapshot=0
        
        self.dt=0.01   # 1 year is kept as starting value for dyanmic dt
        if not self.dynamic_dt:
            self.dt= 0.01

        
        # not in current use but may be useful later
        # self.v_dist=np.empty(0,dtype=np.float64)
        # self.r_dist=np.empty(0,dtype=np.float64)
        
        
        """ Adding Celestial Objects Here """
        # KELPER 35
        # addCelestial(self.planets,self.id_to_sprites,Name="Planet1",radius=2,position=(-0.084,0),mass=0.8877,velocity=(0,-10.1978))
        # addCelestial(self.planets,self.id_to_sprites,Name="Planet2",radius=2,position=(0.092,0),mass=0.8094,velocity=(0,+9.3025))
        
        # SUN & EARTH
        # addCelestial(self.planets,self.id_to_sprites,Name="Sun",radius=8,position=(2.95e-6,0),mass=1,velocity=(0,+1.917e-5))
        # addCelestial(self.planets,self.id_to_sprites,Name="Earth",radius=2,position=(-0.983,0),mass=3.003e-6,velocity=(0,-6.386))

        # Amazing example of 3 body
        # addCelestial(self.planets,self.id_to_sprites,Name="Sun",radius=8,position=(2.95e-6,0),mass=1,velocity=(0,+1.917e-5))
        # addCelestial(self.planets,self.id_to_sprites,Name="Earth",radius=2,position=(0.983,0),mass=3.003e-6,velocity=(0,-6.386))
        # addCelestial(self.planets,self.id_to_spritesName="",radius=5,position=(2,1),mass=6e-1,velocity=(2,-2))
        
        # SUN, Earth, Jupiter

        # addCelestial(self.planets,self.id_to_sprites,Name="Sun",radius=8,position=(0.000,0),mass=1,velocity=(0,0.002643))
        # addCelestial(self.planets,self.id_to_sprites,Name="Earth",radius=1,position=(1,0),mass=3.003e-6,velocity=(0,-6.283))
        # addCelestial(self.planets,self.id_to_sprites,Name="Jupiter",radius=4,position=(5.20,0),mass=9.545e-4,velocity=(0,-2.755))
        # addCelestial(self.planets,self.id_to_sprites,Name="",radius=4,position=(5,-3),mass=1.5,velocity=(-3,-2.5))
        
        #3 Body Stable System to test COLLISION MERGER
        # addCelestial(self.planets,self.id_to_sprites,Name="",radius=5,position=(0,5),mass=2,velocity=(0,0))
        # addCelestial(self.planets,self.id_to_sprites,Name="",radius=5,position=(4.3301,-2.5),mass=2,velocity=(0,0))
        # addCelestial(self.planets,self.id_to_sprites,Name="",radius=5,position=(-4.3301,-2.5),mass=2,velocity=(0,0))
    
        
        #Dyanmic Random System
        for i in range(-30,30):
            for j in range(-30,30):
                addCelestial(self.planets,self.id_to_sprites,"",radius=1,position=(i/2,j/2),mass=random.randrange(1,3),velocity=(random.randrange(0,1),random.randrange(0,1)))
        
        
        #NOTE: self.trial_list should be after adding celestial as it uses len(positions)
        #      if we declare before adding celestial it will be zero length list hence no trails will be stored
        self.trail_list=[[] for _ in range(len(positions))]
        self.scaled_positions = np.zeros_like(positions, dtype=np.float16)
        
        self.adjust_position_barycenter()
        if True:
            self.adjust_velocity_barycenter()
        
        # positions[:, 0] += global_center_x * AU_p_Pix
        # positions[:, 1] += global_center_y * AU_p_Pix
        
        
        # We are adding global center as the input values are from screen's center (0,0) but arcade screen starts from bottem left
        # WE are multiplying AU_p_Pix as the input values are in AU but Global Center in pixels  
        # we are setting the initial positions of sprites here after adjusting barycenter
        # We are sending positions in Celestial in Pixels as it has to be used for display. Positions are in AU
        xs = positions[:, 0]/AU_p_Pix + global_center_x * AU_p_Pix
        ys = positions[:, 1]/AU_p_Pix + global_center_y * AU_p_Pix
        for sprite in self.planets:
            sprite.center_x = xs[sprite.index]
            sprite.center_y = ys[sprite.index]


    def on_draw(self):
        global positions
        self.clear()
        self.camera.use()
        
        color=[arcade.color.RED,arcade.color.BLUE,arcade.color.GREEN,arcade.color.YELLOW,arcade.color.MAGENTA,arcade.color.PINK,arcade.color.PURPLE]
        for i,j in enumerate(self.trail_list):
            arcade.draw_line_strip(j,color[i%len(color)])
            
        # arcade.draw_text(f"DT: {self.dt:.4f} years",
        #                  20, 590,
        #                  arcade.color.WHITE,
        #                  20)
        # arcade.draw_text(f"Time: {self.time_elapsed:.4f} years",
        #                  20, 560,
        #                  arcade.color.WHITE,
        #                  20)
        # y=530
        # for i in masses:
        #     arcade.draw_text(f"Masses: {i:.4e} M☉\n",
        #                      20, y,
        #                      arcade.color.WHITE,
        #                      14)
        #     y-=20
        self.planets.draw()
        
        
                     
    def on_update(self,delta_time):
        if not self.paused:
            
            global positions, velocities, masses, accelerations,store_velocities,store_positions,store_masses
            
            # if self.print_Debug:
            #     if self.frame_count in (1,2,3,4):
                    
            #         print("self.dt: ",self.dt)
            #         print("accelerations 1 : ",accelerations)
            #         print("Position: ",positions)
            #         print("velocity: ",velocities)
            #         print("masses: ",masses,"\n\n")
                    # print(self.Tidal_perturabation(self.planets[0]))
            
            
            # if self.frame_count%20==0:
            #     print(np.sum(masses[:, None] * velocities, axis=0))
        

            """Calculate Acceleration, Velocity and Position"""            
            # Get the current acceleration due to gravity
            accelerations = self.cal_Acceleration()
            
            # Calculate the new positions of Objects
            positions= positions+ velocities*self.dt + 0.5*accelerations*self.dt**2 
            
            #incase of merged objects set their position to zero
            positions[self.deleted]=(0,0)
            
            # Get acceleration after a period of time stamp has passed
            acc_new = self.cal_Acceleration()  # <-- this is a(t+dt)  required to cal Velocity
            
            # Calculates the new velocity
            velocities = velocities + 0.5*(accelerations + acc_new)*self.dt
            
            #incase of merged objects set their velocity to zero
            velocities[self.deleted]=(0,0)
            
            """Collision Detection and Merger"""
            #This is done after updating positions/velocities and storing them + printing so that the effect is seen in next frame itself
            if self.merger:
                self.Merger()

            if self.dynamic_dt:
                self.update_time_period()

            """Setting new positions"""
            xs = positions[:, 0]/AU_p_Pix + global_center_x
            ys = positions[:, 1]/AU_p_Pix + global_center_y
            for sprite in self.planets:

                sprite.center_x = xs[sprite.index]
                sprite.center_y = ys[sprite.index]


            """Storing data for trail: will only have data for specific number of frames, bluring trails"""
            
            self.frame_count+=1
            if self.frame_count% 2== 0:
                # Scale positions (this is fine)
                self.scaled_positions[:, 0] = (positions[:, 0] / AU_p_Pix + global_center_x).astype(np.float16)
                self.scaled_positions[:, 1] = (positions[:, 1] / AU_p_Pix + global_center_y).astype(np.float16)
                # Mask for active (non-removed) particles
                active_mask = masses > 0
                # Append trails only for active particles
                active_indices = np.flatnonzero(active_mask)
                for i in active_indices:
                    self.trail_list[i].append((
                        self.scaled_positions[i, 0],
                        self.scaled_positions[i, 1]
                    ))  # avoid tuple conversion

            """Storing data for analysis"""
            self.time_elapsed+=self.dt
            if self.time_elapsed >= self.next_snapshot :
                # store_positions.append(positions.copy()) # pyright: ignore[reportAttributeAccessIssue]
                # store_velocities.append(velocities.copy())
                # store_masses.append(masses.copy())
                self.next_snapshot += 10


                        
    def on_close(self):
        # global store_positions, store_velocities,store_masses
        # store_positions = np.array(store_positions)
        # try:
        #     T, N, C = store_positions.shape
        #     for i in range(N):
        #         # Save that object's full trajectory as CSV
        #         np.savetxt(f"object_{i}.csv",store_positions[:, i, :],delimiter=",")
        #     # np.savetxt(f"masses.csv",store_masses,delimiter=" ,",fmt="%.2f")
        # except Exception as e:
        #     print("Error saving data:", e)
        super().on_close()

    def on_key_press(self, symbol,modifiers):
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
            
            
    
    def Merger(self):
        
        radii = np.array([sprite.radius for sprite in self.planets])
        pairs, overlaps = wt.spatial_hash_collisions(torch.from_numpy(positions).cuda(), torch.from_numpy(radii).cuda()/50)
        # if pairs.shape[0]!=0:
        #     print(pairs)
        #     print(overlaps)
            

        
    
    def adjust_position_barycenter(self):
        global positions
        sum_of_masses = np.sum(masses)
        center_of_mass = np.sum(positions * masses[:, None], axis=0) / sum_of_masses
        positions -= center_of_mass
        # position=global_center_x*AU_p_Pix)(global_center_y*AU_p_Pix))
        
    def adjust_velocity_barycenter(self):
        global velocities
        sum_of_masses = np.sum(masses)
        center_of_mass = np.sum(velocities * masses[:, None], axis=0) / sum_of_masses
        velocities -= center_of_mass

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
        dt_min_possible=0.009
        dt_max_possible=20
        
        dist=self.get_distance(positions)
        dist[dist==0]=np.inf    #ignore self distance and deleted planets
        min_dist = dist.min()
        speeds = np.linalg.norm(velocities, axis=1)   # speed of each object
        max_velo = speeds.max()  
       # acceleration constraint (REQUIRED for gravity)
        accels = np.linalg.norm(accelerations, axis=1)
        max_accel = accels.max()

        # velocity-based timestep (CFL)
        if max_velo == 0:
            dt_vel = np.inf
        else:
            dt_vel = 0.2 * min_dist / max_velo

        # acceleration-based timestep
        dt_acc = 0.2 * np.sqrt((min_dist / max_accel)) if max_accel > 0 else np.inf

        # final timestep
        self.dt = max( dt_min_possible, min(dt_max_possible, dt_vel, dt_acc) ) / 3    
        
    
        
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
    def Pairwise_Energy(self):
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