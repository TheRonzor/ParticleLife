import plife
from ron_fun import EmotiRon
from ron_qt import (
    qt, 
    qtw, 
    HSliderWithLabels, 
    HSliderLabelAbove,
    LabeledInput
    )

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

mpl.rcParams['toolbar'] = 'None'
plt.style.use('dark_background')

class ParticleLifeApp():

    FORCE_STEPS = 101
    DT_STEPS    = 1001
    FORCE_SCALE = 3
    FIG_SIZE    = (6,6)

    SHOW_MATRIX_LIMIT = 5

    def __init__(self):
        # Initialize world
        self.World = plife.World()
        self.running = False

        # Initialize main display objects
        self.app = qtw.QApplication([])
        self.window = qtw.QWidget()
        self.window.setWindowTitle('Particle Life Controls')
        main_layout = qtw.QVBoxLayout()

        # Force plot ------------------------------------------
        self.init_force_plot(main_layout)

        # Settings for force function -------------------------
        self.force_settings = qtw.QHBoxLayout()
        self.force_settings_left = qtw.QVBoxLayout()
        self.force_settings_right = qtw.QVBoxLayout()
        self.force_settings.addLayout(self.force_settings_left)
        self.force_settings.addLayout(self.force_settings_right)
        main_layout.addLayout(self.force_settings)

        self.r0 = HSliderWithLabels(name   = 'r0', 
                                    vmin   = 0,
                                    vmax   = plife.World.DEFAULT_R[-1],
                                    vdef   = plife.World.DEFAULT_R[0],
                                    vsteps = self.FORCE_STEPS
                                    )
        self.force_settings_left.addLayout(self.r0)

        self.r1 = HSliderWithLabels(name   = 'r1', 
                                    vmin   = 0,
                                    vmax   = plife.World.DEFAULT_R[-1],
                                    vdef   = plife.World.DEFAULT_R[1],
                                    vsteps = self.FORCE_STEPS
                                    )
        self.force_settings_left.addLayout(self.r1)

        self.r2 = HSliderWithLabels(name   = 'r2', 
                                    vmin   = 0,
                                    vmax   = plife.World.DEFAULT_R[-1],
                                    vdef   = plife.World.DEFAULT_R[2],
                                    vsteps = self.FORCE_STEPS
                                    )
        self.force_settings_left.addLayout(self.r2)

        self.f0 = HSliderWithLabels(name    = 'f0', 
                                    vmin    = -1,
                                    vmax    = 0,
                                    vdef    = plife.World.DEFAULT_F[0],
                                    vsteps  = self.FORCE_STEPS
                                    )
        self.force_settings_right.addLayout(self.f0)

        self.f1 = HSliderWithLabels(name   = 'f1', 
                                    vmin   = -1,
                                    vmax   = 1,
                                    vdef   = plife.World.DEFAULT_F[1],
                                    vsteps = self.FORCE_STEPS
                                    )
        self.force_settings_right.addLayout(self.f1)

        self.b = HSliderWithLabels(name   = 'b',
                                   vmin   = 0,
                                   vmax   = 1,
                                   vdef   = plife.World.DEFAULT_B,
                                   vsteps = self.FORCE_STEPS
                                   )
        self.force_settings_right.addLayout(self.b)

        # Additional bindings for the sliders
        self.r0.slider.valueChanged.connect(self.update_force)
        self.r1.slider.valueChanged.connect(self.update_force)
        self.r2.slider.valueChanged.connect(self.update_force)
        self.f0.slider.valueChanged.connect(self.update_force)
        self.f1.slider.valueChanged.connect(self.update_force)
        self.b.slider.valueChanged.connect(self.update_damping)

        # DT
        self.sim_settings = qtw.QHBoxLayout()
        main_layout.addLayout(self.sim_settings)

        self.dt = HSliderWithLabels(name   = 'dt',
                                    vmin   = 0.001,
                                    vmax   = 0.1,
                                    vdef   = self.World.DT,
                                    vsteps = self.DT_STEPS
                                    )
        self.sim_settings.addLayout(self.dt)
        self.dt.slider.valueChanged.connect(self.update_dt)

        # Particle settings
        self.particle_settings = qtw.QHBoxLayout()
        main_layout.addLayout(self.particle_settings)

        self.num_types = LabeledInput('# Types', 40)
        self.particle_settings.addLayout(self.num_types)

        self.num_each = LabeledInput('# Each', 2)
        self.particle_settings.addLayout(self.num_each)

        self.seed = LabeledInput('Seed', self.World.seed)
        self.particle_settings.addLayout(self.seed)

        # Buttons
        self.buttons = qtw.QVBoxLayout()
        main_layout.addLayout(self.buttons)

        self.button_row1 = qtw.QHBoxLayout()
        self.buttons.addLayout(self.button_row1)

        self.go = qtw.QPushButton('Go!')
        self.go.clicked.connect(self.run)
        self.button_row1.addWidget(self.go) 

        self.shake = qtw.QPushButton("Shake 'em up!")
        self.shake.setEnabled(False)
        self.shake.clicked.connect(self.World.shake)
        self.button_row1.addWidget(self.shake)

        self.suck = qtw.QPushButton("Suck 'em in!")
        self.suck.setEnabled(False)
        self.suck.clicked.connect(self.World.suck)
        self.button_row1.addWidget(self.suck)

        self.new_seed = qtw.QPushButton("New Seed")
        self.new_seed.clicked.connect(lambda: self.seed.set_value(np.random.randint(10000)))
        self.button_row1.addWidget(self.new_seed)

        self.button_row2 = qtw.QHBoxLayout()
        self.buttons.addLayout(self.button_row2)

        self.show_interactions = qtw.QPushButton("Show Interactions")
        self.show_interactions.clicked.connect(self.show_interaction_matrix)
        self.show_interactions.setEnabled(False)
        self.button_row2.addWidget(self.show_interactions)

        self.randomize_interactions_button = qtw.QPushButton("Randomize Interactions")
        self.randomize_interactions_button.clicked.connect(self.randomize_interactions)
        self.randomize_interactions_button.setEnabled(False)
        self.button_row2.addWidget(self.randomize_interactions_button)

        #------------------------------------------------------
        self.update_world_settings()
        self.window.setLayout(main_layout)
        self.window.show()
        sys.exit(self.app.exec_())
        return
    
    def init_force_plot(self, layout):
        self.force_fig = plt.figure(figsize=(6,3))
        self.force_ax = self.force_fig.add_subplot(111)
        self.force_ax.set_ylim([-1/self.World.FORCE_DIV, 1/self.World.FORCE_DIV])
        self.force_ax.set_xlim([0,self.World.DEFAULT_R[-1]])
        self.force_ax.set_xticks([0, self.World.DEFAULT_R[-1]])
        self.force_ax.set_yticks([0])
        self.force_ax.set_xlabel('Distance')
        self.force_ax.set_ylabel('Force')
        self.force_fig.tight_layout()
        
        style=':w'
        alpha=0.5

        self.force_curve,   = self.force_ax.plot([],[])
        self.force_r0_up,   = self.force_ax.plot([],[], style, alpha=alpha)
        self.force_r0_over, = self.force_ax.plot([],[], style, alpha=alpha)
        self.force_r1_up,   = self.force_ax.plot([],[], style, alpha=alpha)
        self.force_r1_over, = self.force_ax.plot([],[], style, alpha=alpha)
        self.force_r2_up,   = self.force_ax.plot([],[], style, alpha=alpha)

        self.force_canvas = FigureCanvasQTAgg(self.force_fig)
        layout.addWidget(self.force_canvas)
        return

    
    def update_matrix_from_world(self):
        '''Completely rebuilds the interaction matrix window'''
        if len(self.World.types) > self.SHOW_MATRIX_LIMIT:
            return
        
        #self.matrix_window = qtw.QFrame()
        self.matrix_window = InteractionMatrix(self)
        self.matrix_window.setFrameStyle(qtw.QFrame.Panel | qtw.QFrame.Raised)
        self.matrix_window.setWindowTitle('Interaction Matrix')
        self.matrix_grid = qtw.QGridLayout()
        self.matrix_window.setLayout(self.matrix_grid)

        # This is not ideal! [_]
        hlabels = self.get_matrix_labels()
        vlabels = self.get_matrix_labels()
        
        self.cells={}
        for i in range(self.World.Laws.shape[0]):
            self.matrix_grid.addWidget(hlabels[i], i+1, 0)
            self.matrix_grid.addWidget(vlabels[i], 0, i+1)
            for j in range(self.World.Laws.shape[1]):
                self.cells[(i,j)] = InteractionCell(self.World.Laws[i,j])
                self.cells[(i,j)].cell.slider.valueChanged.connect(
                        partial(self.update_world_from_matrix, i, j)
                        )
                self.matrix_grid.addWidget(self.cells[(i,j)], i+1, j+1)
        return

    def get_matrix_labels(self):
        matrix_labels = {}
        for t, style in self.World.types.items():
            fig, ax = plt.subplots(figsize=(0.5,0.5))
            fig.patch.set_alpha(0)
            
            ax.scatter(0,0,color=style['color'], s=style['size'])
            ax.axis('off')
            ax.axis('square')
            
            canvas = FigureCanvasQTAgg(fig)
            canvas.setStyleSheet("background-color:transparent;")
            matrix_labels[t]= canvas
        return matrix_labels
    
    def update_world_from_matrix(self, i, j):
        self.World.Laws[i,j] = self.cells[i,j].GetValue()
        return
    
    def show_interaction_matrix(self):
        if len(self.World.types) > self.SHOW_MATRIX_LIMIT:
            return
        self.matrix_window.show()
        return

    def randomize_interactions(self):
        self.World.create_random_interactions()
        self.update_matrix_from_world()
        self.show_interaction_matrix()
        return

    def run(self):

        if self.running:
            self.running = False
            self.set_button_states()
            self.World.particles = []
            self.World.types = {}
            try:
                self.ani.event_source.stop()
                plt.close(self.main_fig)
            except:
                pass # Figure was probably closed already
        else:
            self.running = True

            # Create particles and init data
            self.update_world_settings()
            self.create_particles()
            self.World.init_data()

            self.update_matrix_from_world()
            self.show_interaction_matrix()

            # Update buttons
            self.set_button_states()

            # Run the animation
            self.init_main_figure()
            self.ani = FuncAnimation(self.main_fig,
                                     self.update_animation,
                                     init_func = self.first_frame,
                                     blit=False,
                                     interval=10,
                                     cache_frame_data=False,
                                     #save_count=1
                                     )
            self.main_fig.show()
        return
    
    def init_main_figure(self):
        self.main_fig, self.main_ax = plt.subplots(figsize=self.FIG_SIZE)
        self.main_fig.canvas.manager.set_window_title("It's alive!")
        self.main_fig.canvas.mpl_connect('close_event', self.on_figure_close)
        self.main_ax.set_xlim(0, 1)
        self.main_ax.set_ylim(0, 1)
        self.main_ax.axis('off')
        self.main_fig.tight_layout()
        return

    def on_figure_close(self, _):
        self.ani.event_source.stop()
        self.running=False
        self.set_button_states()
        self.World.particles = []
        self.World.types = {}
        return

    def set_button_states(self):
        if self.running:
            self.go.setText('Stop!')
            self.shake.setEnabled(True)
            self.suck.setEnabled(True)
            self.randomize_interactions_button.setEnabled(True)
            if int(self.num_types.get_value()) > 5:
                self.show_interactions.setEnabled(False)
                self.show_interactions.setToolTip("Interaction matrix available when # types is 5 or less")
            else:
                self.show_interactions.setEnabled(True)
        else:
            self.go.setText('Go!')
            self.shake.setEnabled(False)
            self.suck.setEnabled(False)
            self.randomize_interactions_button.setEnabled(False)
            self.show_interactions.setEnabled(False)
            self.show_interactions.setToolTip('')
        return
    
    def first_frame(self):
        self.scat = self.main_ax.scatter(self.World.pos_array[:,0],
                                         self.World.pos_array[:,1],
                                         c = self.World.col_array,
                                         s = self.World.siz_array)
        return self.scat,

    def update_animation(self, _):
        self.World.update_pos()
        self.scat.set_offsets(self.World.pos_array)
        return self.scat,

    def update_world_settings(self):
        self.update_force()
        self.update_damping()
        self.update_dt()
        self.World.update_rng(int(self.seed.get_value()))
        return
    
    def create_particles(self):
        n_types = int(self.num_types.get_value())
        n_each = int(self.num_each.get_value())
        for i in range(n_types):
            self.World.create_particles(i, n_each)
        return
    
    def update_dt(self):
        self.World.DT = self.dt.get_value()
        return
    
    def update_damping(self):
        self.World.damping = self.b.get_value()
        return
    
    def update_force(self):
        r0 = self.r0.get_value()
        r1 = self.r1.get_value()
        r2 = self.r2.get_value()
        f0 = self.f0.get_value()
        f1 = self.f1.get_value()

        self.World.r[0] = r0
        self.World.r[1] = r1
        self.World.r[2] = r2
        self.World.f[0] = f0
        self.World.f[1] = f1

        self.World.update_force()

        d = np.linspace(0,self.World.DEFAULT_R[-1],500)
        f = [self.World.compute_force(di) for di in d]
        self.force_curve.set_data(d,f)

        y0 = self.force_ax.get_ylim()[0]

        self.force_r0_up.set_data([r0, r0], [y0, 0])
        self.force_r0_over.set_data([r0, 0], [0 ,0])
        self.force_r1_up.set_data([r1, r1], [y0, f1/self.World.FORCE_DIV])
        self.force_r1_over.set_data([r1, 0], [f1/self.World.FORCE_DIV, f1/self.World.FORCE_DIV])
        self.force_r2_up.set_data([r2, r2], [y0, 0])

        self.force_canvas.draw()
        return
    
class InteractionCell(qtw.QFrame):
    def __init__(self, value=0):
        super().__init__()

        self.cell = HSliderLabelAbove(vmin = -1,
                                      vmax = 1,
                                      vdef = 0,
                                      vsteps = 201
                                      )
        self.GetValue = self.cell.get_value
        self.SetValue = self.cell.set_value
        self.SetValue(value)

        self.er = EmotiRon((1+self.GetValue())/2, cmap='bwr_r')
        self.fig, self.ax, self.face, self.smile_scat, = self.er.get_plot()
        self.fig.patch.set_alpha(0)
        
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.cell.insertWidget(0, self.canvas)

        self.cell.slider.valueChanged.connect(self.update_face)

        self.setLayout(self.cell)

        self.setFrameStyle(qtw.QFrame.Panel | qtw.QFrame.Raised)
        return
    
    def update_face(self):
        self.er.how_bad = (1+self.GetValue())/2
        self.face.set_color(self.er.get_color())
        self.smile_scat.set_offsets(self.er.make_smile().T)
        self.canvas.draw()
        return
    def close_my_figure(self):
        plt.close(self.fig)
        return
    def closeEvent(self, event):
        print('Closing')
        self.close_my_figure()
        return

class InteractionMatrix(qtw.QFrame):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        return
    def closeEvent(self, event):
        # Explicity close all figures to prevent memory leak
        for c in self.main_app.cells.values():
            c.close_my_figure()
        return

if __name__ == '__main__':
    ParticleLifeApp()