import random  # only needed for the random-system preset below
from collections import deque

import arcade
import numpy as np
import scipy.spatial as ss


AU_p_Pix = 1 / 50      # 1 AU = 50 pixels
G = 39.478             # AU^3 / (M_sun * year^2)  (= 4*pi^2)
softening = 1e-5
global_center_x = 0
global_center_y = 0

# Global state arrays (one row per body; dead bodies stay in the arrays with mass 0)
positions = np.empty((0, 2), dtype=np.float64)       # AU
velocities = np.empty((0, 2), dtype=np.float64)      # AU / year
accelerations = np.empty((0, 2), dtype=np.float64)
masses = np.empty(0, dtype=np.float64)               # solar masses

TRAIL_COLORS = [arcade.color.RED, arcade.color.BLUE, arcade.color.GREEN, arcade.color.YELLOW,
                arcade.color.MAGENTA, arcade.color.PINK, arcade.color.PURPLE]


class Celestial(arcade.SpriteCircle):
    """Base object. r_px is the *physical* radius in pixels (float, grows on merges)."""

    def __init__(self, Name: str, index: int, radius: int, position: tuple[float, float], color):
        super().__init__(radius, color)
        self.name = Name
        self.index = index
        self.r_px = float(radius)
        self.color = color
        self.center_x = position[0]
        self.center_y = position[1]

    def getDistanceFromOther(self, other: "Celestial") -> np.float64:
        return np.float64(np.linalg.norm(positions[self.index] - positions[other.index]))

    def update_texture(self, r_px: float, color) -> None:
        # FIX: radius used to be stored only on the texture, so repeated merges used a stale radius.
        self.r_px = float(r_px)
        r = max(1, int(round(r_px)))
        self.texture = arcade.make_circle_texture(r * 2, color)
        self.width = r * 2
        self.height = r * 2


def addCelestial(spriteList: arcade.SpriteList,
                 sprites_by_id: list,
                 Name: str = "",
                 radius: int = 1,
                 position: tuple[float, float] = (0, 0),
                 mass: float = 0,
                 velocity: tuple[float, float] = (0, 0),
                 acceleration: tuple[float, float] = (0, 0),
                 color=arcade.color.WHITE) -> None:
    """Every object is created through here."""
    global positions, velocities, accelerations, masses

    index = len(masses)
    if Name == "":
        Name = "Planet" + str(index)

    # sprite position is set later in setup(), after the barycenter shift
    sprite = Celestial(Name, index, radius, (0, 0), color)
    spriteList.append(sprite)
    sprites_by_id.append(sprite)

    positions = np.vstack([positions, np.array([position])])
    velocities = np.vstack([velocities, np.array([velocity])])
    accelerations = np.vstack([accelerations, np.array([acceleration])])
    masses = np.append(masses, mass)


class myEnvironment(arcade.Window):
    def __init__(self, title=""):
        global global_center_x, global_center_y

        self.is_fullscreen = False

        screen_w, screen_h = arcade.get_display_size()
        print(f"Resolution: {screen_w}x{screen_h}")

        self.window_w = int(screen_w * 0.8)
        self.window_h = int(screen_h * 0.73)

        super().__init__(self.window_w, self.window_h, title)
        self.set_location(0, 38)

        global_center_x = self.center_x
        global_center_y = self.center_y

    # ------------------------------------------------------------------ setup
    def setup(self):
        arcade.set_background_color(arcade.color.BLACK)

        self.camera = arcade.Camera2D()
        self.dragging = False
        self.paused = False
        self.merger = True
        self.print_Debug = False
        self.dynamic_dt = True
        self.energy_conservation_mode = False
        self.energy_dt = 0.00005
        self.close_encounter_mode = True
        self.close_encounter_threshold = 0.75     # AU: start sub-stepping inside this separation
        self.close_encounter_safety = 0.2
        self.close_encounter_min_dt = 0.000001
        self.close_encounter_max_substeps = 128
        self.collision_scale = 1.0                # merge when dist < scale * (r_i + r_j) (radii in AU)
        self.trail_len = 600                      # points kept per trail (was unbounded)
        self.snapshot_interval = 0.0003           # years between stored snapshots

        self.planets = arcade.SpriteList()
        self.sprites_by_id = []

        self.frame_count = 0
        self.time_elapsed = 0.0
        self.next_snapshot = 0.0

        self.store_positions = []
        self.store_velocities = []
        self.store_masses = []
        self.store_time = []
        self.store_total_energy = []
        self.store_relative_energy_drift = []

        self.dt = 0.01 if self.dynamic_dt else 0.005   # safe startup value; adaptive dt takes over

        """ Adding Celestial Objects Here """
        # KEPLER 35
        # addCelestial(self.planets, self.sprites_by_id, Name="Planet1", radius=2, position=(-0.084, 0), mass=0.8877, velocity=(0, -10.1978))
        # addCelestial(self.planets, self.sprites_by_id, Name="Planet2", radius=2, position=(0.092, 0), mass=0.8094, velocity=(0, +9.3025))

        # SUN & EARTH
        # addCelestial(self.planets, self.sprites_by_id, Name="Sun", radius=8, position=(2.95e-6, 0), mass=1, velocity=(0, +1.917e-5))
        # addCelestial(self.planets, self.sprites_by_id, Name="Earth", radius=2, position=(-0.983, 0), mass=3.003e-6, velocity=(0, -6.386))

        # 3-body example  (FIX: the 3rd line had a typo "self.id_to_spritesName" -> sprites_by_id, Name)
        addCelestial(self.planets, self.sprites_by_id, Name="Sun", radius=8, position=(2.95e-6, 0), mass=1, velocity=(0, +1.917e-5))
        addCelestial(self.planets, self.sprites_by_id, Name="Earth", radius=2, position=(0.983, 0), mass=3.003e-6, velocity=(0, -6.386))
        addCelestial(self.planets, self.sprites_by_id, Name="", radius=5, position=(4, 2), mass=9, velocity=(-5, 5))

        # SUN, Earth, Jupiter
        # addCelestial(self.planets, self.sprites_by_id, Name="Sun", radius=8, position=(0.000, 0), mass=1, velocity=(0, 0.002643))
        # addCelestial(self.planets, self.sprites_by_id, Name="Earth", radius=1, position=(1, 0), mass=3.003e-6, velocity=(0, -6.283))
        # addCelestial(self.planets, self.sprites_by_id, Name="Jupiter", radius=4, position=(5.20, 0), mass=9.545e-4, velocity=(0, -2.755))

        # 3 bodies at rest on a triangle -> they fall together; good test for COLLISION MERGER
        # addCelestial(self.planets, self.sprites_by_id, Name="", radius=5, position=(0, 5), mass=2, velocity=(0, 0))
        # addCelestial(self.planets, self.sprites_by_id, Name="", radius=5, position=(4.3301, -2.5), mass=2, velocity=(0, 0))
        # addCelestial(self.planets, self.sprites_by_id, Name="", radius=5, position=(-4.3301, -2.5), mass=2, velocity=(0, 0))

        # Dynamic random system
        # FIX: random.randrange(0, 1) is always 0, so velocities were always zero. Use uniform.
        # for i in range(-10, 10):
        #     for j in range(-10, 10):
        #         addCelestial(self.planets, self.sprites_by_id, "", radius=3, position=(i * 1.1, j * 1.1),
        #                      mass=random.uniform(1, 3), velocity=(random.uniform(-1, 1), random.uniform(-1, 1)))

        # NOTE: everything below must come AFTER adding celestials (it uses len(positions))
        self.radii_px = np.array([s.r_px for s in self.sprites_by_id], dtype=np.float64)
        self.trail_list = [deque(maxlen=self.trail_len) for _ in range(len(positions))]

        self.adjust_position_barycenter()
        self.adjust_velocity_barycenter()

        # initial acceleration + adaptive dt before the first integration step
        accelerations[masses != 0] = self.cal_Acceleration()
        if self.dynamic_dt:
            self.update_time_period(accelerations)

        # FIX: initial sprite placement used `global_center_x * AU_p_Pix` while on_update uses
        # `global_center_x` -> sprites jumped on the first frame. Same formula everywhere now.
        self._sync_sprites()

        self.E0 = float(np.sum(self.Total_energy_per_body()))

    # ------------------------------------------------------------------ drawing
    def on_draw(self):
        self.clear()
        self.camera.use()

        for k, trail in enumerate(self.trail_list):
            if len(trail) > 1:
                arcade.draw_line_strip(list(trail), TRAIL_COLORS[k % len(TRAIL_COLORS)])

        self.planets.draw()  # type: ignore

    def _sync_sprites(self):
        xs = positions[:, 0] / AU_p_Pix + global_center_x
        ys = positions[:, 1] / AU_p_Pix + global_center_y
        for sprite in self.planets:
            sprite.center_x = xs[sprite.index]
            sprite.center_y = ys[sprite.index]

    # ------------------------------------------------------------------ update loop
    def on_update(self, delta_time):
        if self.paused:
            return

        # FIX: time_elapsed used to add self.dt AFTER update_time_period() had already changed it,
        # so the clock advanced by the wrong step. Remember the step actually taken.
        dt_used = self.dt
        self.advance_physics(dt_used)

        if self.energy_conservation_mode:
            self.dt = self.energy_dt
        elif self.dynamic_dt:
            self.update_time_period(accelerations)

        self._sync_sprites()

        self.frame_count += 1
        if self.frame_count % 2 == 0:
            act = np.flatnonzero(masses > 0)
            px = positions[:, 0] / AU_p_Pix + global_center_x
            py = positions[:, 1] / AU_p_Pix + global_center_y
            for i in act:
                self.trail_list[i].append((float(px[i]), float(py[i])))

        self.time_elapsed += dt_used
        if self.time_elapsed >= self.next_snapshot:
            self._record_snapshot()
            # FIX: was `+= 0.0003`, which lags behind and stores *every* frame once dt > 0.0003
            self.next_snapshot = self.time_elapsed + self.snapshot_interval

        if self.print_Debug and self.frame_count % 20 == 0:
            print(f"Dynamics: {self.calculate_dynamics()} ")



    def _record_snapshot(self):
        self.store_positions.append(positions.copy())
        self.store_velocities.append(velocities.copy())
        self.store_masses.append(masses.copy())
        self.store_time.append(self.time_elapsed)
        e = float(np.sum(self.Total_energy_per_body()))      # computed once (was computed twice)
        self.store_total_energy.append(e)
        self.store_relative_energy_drift.append((e - self.E0) / abs(self.E0) if self.E0 != 0 else 0.0)

    def on_close(self):
        try:
            T = len(self.store_positions)
            if T > 0:
                N = self.store_positions[0].shape[0]
                print("T :", T, "   N:", N)
                pos = np.concatenate(self.store_positions)          # (T*N, 2)
                vel = np.concatenate(self.store_velocities)
                mass = np.concatenate(self.store_masses)
                t = np.repeat(np.asarray(self.store_time), N)
                ids = np.tile(np.arange(N), T)
                table = np.column_stack([t, ids, pos, vel, mass])
                # FIX: was opened with "a" and wrote a header every run (duplicate headers); "w" now.
                np.savetxt("simulation_data.csv", table, delimiter=",",
                           header="time,object_id,x,y,vx,vy,mass", comments="")
                energy = np.column_stack([self.store_time, self.store_total_energy,
                                          self.store_relative_energy_drift])
                np.savetxt("simulation_energy.csv", energy, delimiter=",",
                           header="time,total_energy,relative_energy_drift", comments="")
        except Exception as e:
            print("Error saving data:", e)

        super().on_close()

    # ------------------------------------------------------------------ input
    def on_key_press(self, symbol, modifiers):
        global global_center_x, global_center_y
        if symbol == arcade.key.F:
            self.is_fullscreen = not self.is_fullscreen
            self.set_fullscreen(self.is_fullscreen)

            if not self.is_fullscreen:
                self.set_size(self.window_w, self.window_h)
                self.set_location(0, 38)

            global_center_x = self.center_x
            global_center_y = self.center_y

        elif symbol == arcade.key.SPACE:
            self.paused = not self.paused

        elif symbol == arcade.key.P:
            self.print_Debug = not self.print_Debug

        elif symbol == arcade.key.E:
            self.energy_conservation_mode = not self.energy_conservation_mode
            mode = "ON" if self.energy_conservation_mode else "OFF"
            print(f"Energy conservation mode: {mode} (fixed dt = {self.energy_dt})")

    def on_mouse_press(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            self.dragging = True

    def on_mouse_release(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            self.dragging = False

    def on_mouse_motion(self, x, y, dx, dy):
        if self.dragging:
            # FIX: divide by zoom so the scene follows the cursor at any zoom level
            z = self.camera.zoom
            self.camera.position = (self.camera.position[0] - dx / z,
                                    self.camera.position[1] - dy / z)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        zoom_speed = 0.1
        if scroll_y > 0:
            self.camera.zoom *= (1 + zoom_speed)
        elif scroll_y < 0:
            self.camera.zoom /= (1 + zoom_speed)

    # ------------------------------------------------------------------ physics
    def advance_physics(self, step_dt: float):
        """Velocity-Verlet with optional close-encounter sub-stepping."""
        substeps = 1
        if self.close_encounter_mode and not self.energy_conservation_mode:
            close_dt = self.close_encounter_dt()
            if close_dt is not None and close_dt < step_dt:
                substeps = min(self.close_encounter_max_substeps,
                               max(2, int(np.ceil(step_dt / close_dt))))
        h = step_dt / substeps

        acc_dirty = False   # becomes True when a merge invalidates the stored accelerations
        for _ in range(substeps):
            act = np.flatnonzero(masses != 0)
            if act.size == 0:
                break

            # PERF: accelerations from the end of the previous sub-step are still valid
            # (same positions, same masses). The old code recomputed them every sub-step,
            # i.e. 2 force evaluations per step instead of 1.
            if acc_dirty:
                accelerations[act] = self.cal_Acceleration(act)
                acc_dirty = False

            positions[act] += velocities[act] * h + 0.5 * accelerations[act] * h * h
            acc_new = self.cal_Acceleration(act)
            velocities[act] += 0.5 * (accelerations[act] + acc_new) * h
            accelerations[act] = acc_new

            if self.merger and self.merge_collisions(act):
                acc_dirty = True

        if acc_dirty:   # make sure accelerations match the final state (used for adaptive dt)
            act = np.flatnonzero(masses != 0)
            if act.size:
                accelerations[act] = self.cal_Acceleration(act)

    def merge_collisions(self, act: np.ndarray) -> bool:
        """
        Perfectly-inelastic merging (conserves mass and momentum, NOT energy).
        Collision test uses each body's own radius (radii_px * AU_p_Pix), so bodies that grow by
        merging also get a bigger collision radius. Previously a fixed 0.275 AU was used.
        """
        if act.size < 2:
            return False

        pos = positions[act]
        r_au = self.radii_px[act] * AU_p_Pix * self.collision_scale
        tree = ss.cKDTree(pos)
        pairs = tree.query_pairs(r=2.0 * float(r_au.max()), output_type="ndarray")
        if pairs.size == 0:
            return False

        d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
        hit = d < (r_au[pairs[:, 0]] + r_au[pairs[:, 1]])
        if not hit.any():
            return False
        pairs, d = pairs[hit], d[hit]
        pairs = pairs[np.argsort(d)]                 # closest pairs first
        gi, gj = act[pairs[:, 0]].tolist(), act[pairs[:, 1]].tolist()   # global indices

        dead_sprites = []
        merged = False
        for a, b in zip(gi, gj):
            if masses[a] == 0 or masses[b] == 0:     # already absorbed earlier in this loop
                continue

            w, l = (a, b) if masses[a] >= masses[b] else (b, a)
            mw, ml = masses[w], masses[l]
            M = mw + ml

            velocities[w] = (mw * velocities[w] + ml * velocities[l]) / M
            positions[w] = (mw * positions[w] + ml * positions[l]) / M
            masses[w] = M

            new_r = (self.radii_px[w] ** 3 + self.radii_px[l] ** 3) ** (1 / 3)   # volume additive
            self.radii_px[w] = new_r
            big = self.sprites_by_id[w]
            big.update_texture(new_r, big.color)

            masses[l] = 0.0
            velocities[l] = 0.0
            positions[l] = 0.0
            accelerations[l] = 0.0
            dead_sprites.append(self.sprites_by_id[l])
            self.sprites_by_id[l] = None
            merged = True
            print(f"MERGED: {l} -> {w}")

        for s in dead_sprites:
            self.planets.remove(s)
        return merged

    def _closest_pair(self, act: np.ndarray):
        """Closest pair of live bodies via KD-tree: O(N log N) instead of an N x N x 2 tensor."""
        if act.size < 2:
            return None
        p = positions[act]
        dist, nn = ss.cKDTree(p).query(p, k=2)
        i = int(np.argmin(dist[:, 1]))
        return i, int(nn[i, 1]), float(dist[i, 1])     # indices local to `act`

    def _close_dt_from_pair(self, act, pair, accel_source):
        if pair is None:
            return None
        li, lj, min_dist = pair
        if min_dist >= self.close_encounter_threshold:
            return None
        i, j = act[li], act[lj]
        rel_speed = np.linalg.norm(velocities[i] - velocities[j])
        rel_accel = np.linalg.norm(accel_source[i] - accel_source[j])

        cands = []
        if rel_speed > 0:
            cands.append(self.close_encounter_safety * min_dist / rel_speed)
        if rel_accel > 0:
            cands.append(self.close_encounter_safety * np.sqrt(min_dist / rel_accel))
        if not cands:
            return self.close_encounter_min_dt
        return max(self.close_encounter_min_dt, min(cands))

    def close_encounter_dt(self, accel_source: np.ndarray | None = None):
        act = np.flatnonzero(masses != 0)
        if act.size < 2:
            return None
        if accel_source is None:
            accel_source = accelerations
        return self._close_dt_from_pair(act, self._closest_pair(act), accel_source)

    def update_time_period(self, accel_source: np.ndarray | None = None) -> None:
        """Adaptive dt: min(CFL on velocity, acceleration criterion, close-encounter criterion)."""
        dt_min_possible = 0.005
        dt_max_possible = 20

        act = np.flatnonzero(masses != 0)
        if act.size == 0:
            return
        if accel_source is None:
            accel_source = accelerations

        pair = self._closest_pair(act)
        min_dist = np.inf if pair is None else max(pair[2], 1e-12)

        max_velo = np.linalg.norm(velocities[act], axis=1).max()
        max_accel = np.linalg.norm(accel_source[act], axis=1).max()

        dt_vel = np.inf if max_velo == 0 else 0.2 * min_dist / max_velo
        dt_acc = 0.2 * np.sqrt(min_dist / max_accel) if max_accel > 0 else np.inf

        cands = [dt_max_possible, dt_vel, dt_acc]
        close_dt = self._close_dt_from_pair(act, pair, accel_source)
        if close_dt is not None:
            cands.append(close_dt)

        target_dt = max(dt_min_possible, min(cands)) / 3
        # avoid sudden large increases in dt after close encounters
        self.dt = min(target_dt, self.dt * 1.2)

    def adjust_position_barycenter(self):
        global positions
        total = np.sum(masses, dtype=np.float64)
        com = np.sum(positions * masses[:, None], axis=0, dtype=np.float64) / total
        positions -= com

    def adjust_velocity_barycenter(self):
        global velocities
        total = np.sum(masses, dtype=np.float64)
        com = np.sum(velocities * masses[:, None], axis=0, dtype=np.float64) / total
        velocities -= com

    def cal_Acceleration(self, act: np.ndarray | None = None):
        """
        Plummer-softened gravity on the live bodies.
        acc_i = -G * sum_j m_j (r_i - r_j) / (|r_i - r_j|^2 + eps^2)^(3/2)
        """
        if act is None:
            act = np.flatnonzero(masses != 0)
        p = positions[act]
        m = masses[act]

        diff = p[:, None, :] - p[None, :, :]                        # (n, n, 2)
        r2 = np.einsum("ijk,ijk->ij", diff, diff) + softening ** 2
        w = r2 ** -1.5 * m[None, :]                                 # i==j contributes diff=0
        return -G * np.einsum("ijk,ij->ik", diff, w)

    # ------------------------------------------------------------------ diagnostics
    def get_distance(self, coordinate_list: np.ndarray):
        """
        Pairwise euclidean distance for FULL-length arrays (rows of dead bodies are zeroed).
        FIX: the old version indexed with a list of *global* dead ids even when given a compacted
        array (close_encounter_dt), which raised IndexError / zeroed the wrong rows after a merge.
        """
        diffs = coordinate_list[:, None, :] - coordinate_list[None, :, :]
        out = np.linalg.norm(diffs, axis=2)
        dead = masses == 0
        out[dead, :] = 0
        out[:, dead] = 0
        return out

    def Orbital_energy(self):
        v = self.get_distance(velocities)
        r = self.get_distance(positions)

        active = (masses[:, None] != 0) & (masses[None, :] != 0)
        np.fill_diagonal(active, False)

        result = np.full_like(r, np.nan, dtype=np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            result[active] = (
                0.5 * v[active] ** 2
                - G * (masses[:, None] + masses[None, :])[active] / r[active]
            )

        return result

    def angular_momentum(self):
        return np.cross(positions, velocities) * masses

    def Pairwise_Energy(self):
        mul = masses[:, None] * masses[None, :]
        add = masses[:, None] + masses[None, :]
        v2 = self.get_distance(velocities) ** 2
        r = self.get_distance(positions)

        active = (masses[:, None] != 0) & (masses[None, :] != 0)
        np.fill_diagonal(active, False)

        result = np.full_like(r, np.nan, dtype=np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            result[active] = (
                0.5 * (mul[active] / add[active]) * v2[active]
                - G * mul[active] / r[active]
            )

        return result

    def Tidal_perturabation(self, primary: Celestial):
        r_dist = np.linalg.norm(positions - positions[primary.index], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            n = (
                    masses[None, :] / masses[primary.index]
                ) * (
                    r_dist[:, None] / r_dist[None, :]
                ) ** 3
        n[primary.index] = np.nan
        n[:, primary.index] = np.nan
        np.fill_diagonal(n, np.nan)
        return n

    def kinetic_energy_per_body(self):
        return 0.5 * masses * np.sum(velocities ** 2, axis=1)       # dead bodies have m = 0

    def potential_energy_per_body(self):
        """PERF: vectorised (was an O(N^2) pure-Python double loop, run every snapshot)."""
        V = np.zeros_like(masses, dtype=np.float64)
        act = np.flatnonzero(masses != 0)
        if act.size < 2:
            return V
        m = masses[act]
        p = positions[act]
        d = p[:, None, :] - p[None, :, :]
        r = np.sqrt(np.einsum("ijk,ijk->ij", d, d) + softening ** 2)   # same softening as the force
        U = -G * m[:, None] * m[None, :] / r
        np.fill_diagonal(U, 0.0)
        V[act] = 0.5 * U.sum(axis=1)       # each pair's energy split half/half between its bodies
        return V

    def Total_energy_per_body(self):
        """Summed over all bodies this is the total energy of the system."""
        return self.kinetic_energy_per_body() + self.potential_energy_per_body()

    def calculate_dynamics(self):
        pairwise = self.Pairwise_Energy()
        N = pairwise.shape[0]
        visited = [False] * N
        groups = []

        for i in range(N):
            if visited[i]:
                continue
            stack = [i]
            group = {i}
            visited[i] = True
            while stack:
                a = stack.pop()
                for b in range(N):
                    if not visited[b] and pairwise[a, b] < 0:
                        visited[b] = True
                        stack.append(b)
                        group.add(b)
            groups.append(sorted(group))
        print("Gravitationally Bound Groups:", groups)


if __name__ == "__main__":
    # FIX: window creation used to run at import time, so the module could not be imported/tested.
    Env = myEnvironment(title="First")
    Env.setup()
    arcade.run()