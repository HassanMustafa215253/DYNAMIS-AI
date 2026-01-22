import arcade
import arcade.gui


class MyEnv(arcade.Window):
    def __init__(self,title=""):

        self.is_fullscreen=False
        
        screen_w, screen_h = arcade.get_display_size()
        print(f"Resolution: {screen_w}x{screen_h}")

        self.window_w = int(screen_w * 0.8)
        self.window_h = int(screen_h * 0.73)

        super().__init__(self.window_w, self.window_h,title)
        self.set_location(0, 38)
          
    def setup(self):
        arcade.set_background_color(arcade.color.BLACK)
        
        if True:
            self.camera = arcade.Camera2D()
            self.dragging=False
            self.a=arcade.SpriteList()
            b=arcade.SpriteCircle(50,arcade.color.BLUE_BELL)
            self.a.append(b)
            b.center_x=self.center_x
            b.center_y=self.center_y        
            self.dt=1  
        
        self.UI= arcade.gui.UIManager()
        self.UI.enable()
        self.UI.add(self.PerminentUI())
        
    def PerminentUI(self):
        anchor=arcade.gui.UIAnchorLayout()
        self.PUIDeltaDisplay=arcade.gui.UILabel(f"Time Period (dt) : {self.dt:.2f}",text_color= arcade.color.WHITE,font_size=20)
        anchor.add(self.PUIDeltaDisplay,anchor_x="right",anchor_y="top")
        return anchor
    
    def on_draw(self):
        self.clear()
        self.camera.use()
        self.a.draw()
        self.UI.draw()
        
    def on_update(self,delta_time):
        self.dt += 0.01
        self.updatePerUI()
        
        self.a[0].center_x+=0.5
        self.a[0].center_y+=0.5
        
    def updatePerUI(self):
        self.PUIDeltaDisplay.text = f"dt = {self.dt:.2f}"

    
    def RunningUI(self):
        pass
    
    def EditUI(self):
        pass
    
    
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
    
        
Env=MyEnv(title="First")
Env.setup()

arcade.run()