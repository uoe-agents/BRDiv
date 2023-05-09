"""
2D rendering of the level based foraging domain
"""

import math
import os
import sys

import numpy as np
import math
import six
from gym import error

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = 50
        self.icon_size = 20

        self.width = self.cols * self.grid_size + 1
        self.height = self.rows * self.grid_size + 1
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        script_dir = os.path.dirname(__file__)

        pyglet.resource.path = [os.path.join(script_dir, "icons")]
        pyglet.resource.reindex()

        self.img_agent = pyglet.resource.image("agent.png")
        self.img_prey = pyglet.resource.image("prey.png")
        self.img_longsword = pyglet.resource.image("sword.png")
        self.img_greatsword = pyglet.resource.image("greatsword.png")
        self.img_bow = pyglet.resource.image("bow.png")
        self.img_healing_staff = pyglet.resource.image("healingstaff.png")
        self.img_shield = pyglet.resource.image("shield.png")
        self.img_chain = pyglet.resource.image("chains.png")
        self.img_scroll = pyglet.resource.image("scroll.png")
        self.img_spellbook = pyglet.resource.image("spellbook.png")

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def render(self, env, return_rgb_array=False):
        glClearColor(0, 0, 0, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_players_and_items(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,
                        self.grid_size * r,
                        self.grid_size * self.cols,
                        self.grid_size * r,
                    ),
                ),
                ("c3B", (*_WHITE, *_WHITE)),
            )
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        self.grid_size * c,
                        0,
                        self.grid_size * c,
                        self.grid_size * self.rows,
                    ),
                ),
                ("c3B", (*_WHITE, *_WHITE)),
            )
        batch.draw()

    def _draw_players_and_items(self, env):
        players = []
        player_coordinates = []
        items = []
        batch = pyglet.graphics.Batch()

        for player in env.players:
            col = player.position[0]
            row = player.position[1]

            player_coordinates.append((col, row))
            if player.agent_type == "predator":
                players.append(
                    pyglet.sprite.Sprite(
                        self.img_agent,
                        self.grid_size * col,
                        self.grid_size * row,
                        batch=batch,
                    )
                )
            else:
                players.append(
                    pyglet.sprite.Sprite(
                        self.img_prey,
                        self.grid_size * col,
                        self.grid_size * row+0.5*self.grid_size,
                        batch=batch,
                    )
                )
        for p in players:
            p.update(scale=self.grid_size / p.width)

        non_zero_items_y, non_zero_items_x = np.where(env.field != -1)
        for y_coord, x_coord in zip(non_zero_items_y, non_zero_items_x):
            if not (x_coord, y_coord) in player_coordinates:
                item_type = env.field[y_coord][x_coord]
                sprite = None

                # If item is sword
                if item_type == 0:
                    sprite = self.img_longsword
                elif item_type == 1:
                    sprite = self.img_greatsword
                elif item_type == 2:
                    sprite = self.img_bow
                elif item_type == 3:
                    sprite = self.img_healing_staff
                elif item_type == 4:
                    sprite = self.img_shield
                elif item_type == 5:
                    sprite = self.img_chain
                elif item_type == 6:
                    sprite = self.img_scroll
                elif item_type == 7:
                    sprite = self.img_spellbook

                items.append(
                    pyglet.sprite.Sprite(
                        sprite,
                        self.grid_size * x_coord,
                        self.grid_size * y_coord,
                        batch=batch,
                    )
                )

        for i in items:
            i.update(scale=self.grid_size / i.height)

        batch.draw()
        for p in env.players:
            self._draw_badge(*p.position, p.hp, p.item)

    def _draw_badge(self, col, row, level, item):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * self.grid_size + (3 / 4) * self.grid_size
        badge_y = row * self.grid_size + (1 / 4) * self.grid_size

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_BLACK)
        circle.draw(GL_POLYGON)
        glColor3ub(*_WHITE)
        circle.draw(GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(level),
            font_name="Times New Roman",
            font_size=12,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
        )
        label.draw()


        if item != -1:
            batch = pyglet.graphics.Batch()
            sprite = None
            if item == 0:
                sprite=self.img_longsword
            elif item == 1:
                sprite = self.img_greatsword
            elif item == 2:
                sprite = self.img_bow
            elif item == 3:
                sprite = self.img_healing_staff
            elif item == 4:
                sprite = self.img_shield
            elif item == 5:
                sprite = self.img_chain
            elif item == 6:
                sprite = self.img_scroll
            elif item== 7:
                sprite = self.img_spellbook

            p = pyglet.sprite.Sprite(
                sprite,
                self.grid_size * col,
                self.grid_size * row,
                batch=batch,
            )
            p.update(scale=0.5 * self.grid_size / p.width)
            # (1 / 4) * self.grid_size
            # (3 / 4) * self.grid_size
            batch.draw()
